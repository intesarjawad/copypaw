"""Command-line entry point."""

import logging
import re
import shutil
import sys
import tempfile
import traceback
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from typer.core import TyperGroup

from copypaw.animate.export import (
    drawing_facing,
    load_clip_frames,
    side_by_side,
    write_drawing,
    write_footfall,
)
from copypaw.animate.render import RenderResult, render_character, resample_motion
from copypaw.app.selftest import format_report, run_self_test
from copypaw.app.server import STATIC_DIR, LocalApp, Pipeline
from copypaw.capture.pipeline import capture_motion, load_stages
from copypaw.capture.video import read_frames
from copypaw.doctor import report
from copypaw.errors import CopypawError, InternalError
from copypaw.evaluate.fidelity import (
    CalibrationError,
    DrawingJoints,
    drawing_contacts,
    evaluate,
    load_calibration,
    write_fidelity,
)
from copypaw.models import Mode, ensure_models
from copypaw.motion.motion_file import MotionFile
from copypaw.rig.char_cfg import write_character
from copypaw.rig.edits import (
    joints_to_json,
    load_drawing,
    load_joints,
    rig_drawing,
    write_edits,
)


class OutputDirectoryError(CopypawError):
    """The directory for a command's results could not be created."""

    codes = ("output_directory",)


_LEGS_GUESSED_WARNING = (
    "Warning: no four or two separate legs were found (a tail or wing that reaches the "
    "ground counts as a leg), so the leg joints are a guess and may sit outside the drawing. "
    "Save the printed joints to a file, move the leg joints onto the legs and rerun with "
    "--joints FILE."
)


class _ErrorReportingGroup(TyperGroup):
    """Print every failure as the two-line `error:` and `next:` report."""

    # The context class lives in Typer's private click copy, so it is typed loosely.
    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except (typer.TyperException, typer.Exit, typer.Abort):
            raise
        except Exception as error:
            known = error if isinstance(error, CopypawError) else InternalError()
            typer.echo(f"error: {_one_line(known.what)}", err=True)
            typer.echo(f"next:  {_one_line(known.next)}", err=True)
            if ctx.params.get("debug"):
                traceback.print_exception(error, file=sys.stderr)
            raise typer.Exit(code=known.exit_code) from error


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


app = typer.Typer(no_args_is_help=True, cls=_ErrorReportingGroup)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models", help="Fetch verified model weights.")


@app.callback()
def root(
    debug: Annotated[
        bool, typer.Option("--debug", help="Show the traceback when a command fails.")
    ] = False,
) -> None:
    """Track a pet clip and animate a drawing, in the local app or step by step."""
    del debug  # Read from the context by the error report.


def _make_output_directory(out: Path) -> None:
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputDirectoryError(path=out, reason=error) from error


@app.command()
def doctor() -> None:
    """Report local runtime and model cache status."""
    typer.echo(report())


@models_app.command("fetch")
def fetch_models(
    mode: Annotated[Mode, typer.Option(help="Detector and pose model pair.")] = Mode.BALANCED,
) -> None:
    """Download missing weights and verify every model's SHA-256."""

    def progress(filename: str, downloaded: int, total: int) -> None:
        typer.echo(f"{filename}: {downloaded}/{total} bytes", err=True)

    paths = ensure_models(mode, progress)
    for key, path in paths.items():
        typer.echo(f"Verified {key}: {path}")


@app.command()
def capture(
    clip: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Side-view pet clip."),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", file_okay=False, help="Directory for the motion files.")
    ],
    mode: Annotated[Mode, typer.Option(help="Detector and pose model pair.")] = Mode.BALANCED,
    flip_test: Annotated[
        bool, typer.Option(help="Average pose heatmaps over a mirrored pass.")
    ] = True,
) -> None:
    """Extract motion.npz, motion.bvh and quality.json from a pet clip."""

    def progress(filename: str, downloaded: int, total: int) -> None:
        typer.echo(f"{filename}: {downloaded}/{total} bytes", err=True)

    _make_output_directory(out)
    frames = read_frames(clip)
    models = ensure_models(mode, progress)
    detector, pose = load_stages(mode, flip_test, models)
    quality = capture_motion(frames, clip.name, out, detector=detector, pose=pose)
    for warning in quality.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(f"Wrote motion.npz, motion.bvh and quality.json to {out}")


def _drawing_joints(rendered: RenderResult) -> DrawingJoints | None:
    """Use the unfiltered render joints when the engine recorded joint names."""
    if rendered.joint_names is None:
        return None
    unfiltered = rendered.unfiltered_joints_per_frame
    return DrawingJoints(
        joints=rendered.joints_per_frame,
        joints_unfiltered=rendered.joints_per_frame if unfiltered is None else unfiltered,
        names=rendered.joint_names,
        fps=rendered.fps,
    )


@app.command()
def animate(  # noqa: PLR0917
    char_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, help="Drawing directory."),
    ],
    motion: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Captured motion archive."),
    ],
    clip: Annotated[
        Path,
        typer.Option("--clip", exists=True, dir_okay=False, readable=True, help="Source pet clip."),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", file_okay=False, help="Directory for the animations.")
    ],
    mode: Annotated[
        Literal["treadmill", "travel"],
        typer.Option(help="Keep the root in place or let it travel."),
    ] = "treadmill",
    keep_facing: Annotated[
        bool,
        typer.Option("--keep-facing", help="Do not flip a drawing that faces the other way."),
    ] = False,
    credit: Annotated[
        bool,
        typer.Option("--credit/--no-credit", help="Draw the project credit under each frame."),
    ] = True,
) -> None:
    """Animate a drawing from captured motion beside its source clip."""
    _make_output_directory(out)
    captured = MotionFile.load(motion)
    clip_frames = load_clip_frames(clip)
    # The engine's configs and baked BVH are scratch files; only its joints are a result.
    with tempfile.TemporaryDirectory(prefix="copypaw-render-") as work:
        rendered = render_character(char_dir, captured, mode, Path(work))
        shutil.copyfile(Path(work) / "render_joints.npz", out / "render_joints.npz")
    drawing = _drawing_joints(rendered)
    drawing_stance = None
    if drawing is not None:
        try:
            percentile = load_calibration()
        except CalibrationError:
            percentile = None
        write_fidelity(
            out / "fidelity.json",
            evaluate(captured, drawing),
            percentile=percentile,
        )
        # The strip shows the motion the viewer sees: the animation after foot planting.
        drawing_stance = drawing_contacts(replace(drawing, joints_unfiltered=drawing.joints))
    write_drawing(
        rendered.frames_rgba,
        rendered.fps,
        out / "animation.mp4",
        out / "animation.gif",
    )
    pet_contacts = resample_motion(captured).contacts
    side_by_side(
        clip_frames,
        rendered.frames_rgba,
        rendered.fps,
        captured.facing,
        drawing_facing(char_dir),
        keep_facing,
        credit,
        out / "side_by_side.mp4",
        out / "side_by_side.gif",
        contacts=pet_contacts,
        drawing_contacts=drawing_stance,
    )
    write_footfall(
        out / "footfall.json",
        pet_contacts,
        drawing_stance,
        min(len(clip_frames), len(rendered.frames_rgba)),
    )
    typer.echo(f"Wrote animation and side-by-side videos to {out}")


@app.command()
def rig(
    drawing: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Drawing of one pet."),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", file_okay=False, help="Directory for the rigged drawing.")
    ],
    joints: Annotated[
        Path | None,
        typer.Option(
            "--joints",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Joint positions to use instead of the snapped ones.",
        ),
    ] = None,
) -> None:
    """Rig a drawing and print its 16 joints as JSON."""
    _make_output_directory(out)
    image = load_drawing(drawing)
    rigged = rig_drawing(image)
    final = rigged.snapped if joints is None else load_joints(joints, rigged.snapped)
    if joints is None and not rigged.legs_snapped:
        typer.echo(_LEGS_GUESSED_WARNING, err=True)
    write_character(image, final, out)
    if joints is not None:
        write_edits(out / "rig_edits.json", rigged.snapped, final, rigged.box_height)
    typer.echo(joints_to_json(final))
    typer.echo(f"Wrote the rigged drawing to {out}", err=True)


@app.command("app")
def serve_app(
    port: Annotated[
        int, typer.Option(min=0, max=65535, help="Port on 127.0.0.1; 0 picks a free one.")
    ] = 0,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Print the address without opening it.")
    ] = False,
    self_test: Annotated[
        bool,
        typer.Option(
            "--self-test",
            help="Run the packaged sample pair through the app, print the timings and exit.",
        ),
    ] = False,
    no_render: Annotated[
        bool,
        typer.Option("--no-render", help="With --self-test, skip the steps that render."),
    ] = False,
) -> None:
    """Open the local app in a browser and serve it until Ctrl+C."""
    pipeline = Pipeline(capture=capture, animate=animate)
    if self_test:
        typer.echo("Running the sample pair; a first run downloads the pose models.", err=True)
        report = run_self_test(pipeline, render=not no_render)
        typer.echo(format_report(report))
        raise typer.Exit(code=0 if report.passed else 1)
    if no_render:
        raise typer.BadParameter("--no-render needs --self-test")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    local = LocalApp(pipeline, static_dir=STATIC_DIR, port=port)
    try:
        local.start()
        typer.echo(f"Copypaw is running at {local.url}")
        typer.echo("Press Ctrl+C to stop.")
        if not no_browser:
            webbrowser.open(local.url)
        local.wait()
    except KeyboardInterrupt:
        pass
    finally:
        local.close()


def main() -> None:
    """Run the command-line interface."""
    app()

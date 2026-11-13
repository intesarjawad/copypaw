import type { ComputeBackend, GalleryMotion, MotionChoice } from "../backend";
import { STRINGS } from "../strings";
import { el, showError } from "./dom";

export interface MotionStep {
  readonly element: HTMLElement;
  choice(): MotionChoice | null;
  showGallery(gallery: readonly GalleryMotion[]): void;
  /** Offer the clip input only when the backend can track a clip. */
  allowClips(allowed: boolean): void;
}

export function motionStep(backend: ComputeBackend, onChange: () => void): MotionStep {
  const input = el("input", {
    type: "file",
    id: "clip-input",
    accept: "video/*",
    capture: "environment",
  });
  const clipLabel = el("label", { for: "clip-input" }, STRINGS.chooseClip);
  const status = el("p", { role: "status" });
  const errors = el("div", { role: "alert" });
  const gallery = el("fieldset", { class: "gallery" }, el("legend", {}, STRINGS.galleryLegend));
  gallery.hidden = true;
  const element = el(
    "section",
    { class: "step motion", "aria-labelledby": "motion-heading" },
    el("h3", { id: "motion-heading" }, STRINGS.motionHeading),
    clipLabel,
    input,
    status,
    errors,
    gallery,
  );

  let choice: MotionChoice | null = null;
  const radios: HTMLInputElement[] = [];
  let motionIds: readonly string[] = [];
  let clipsAllowed = true;
  let selection = 0;

  function settle(): void {
    const [id] = motionIds;
    const single = id !== undefined && motionIds.length === 1 && !clipsAllowed;
    gallery.hidden = motionIds.length === 0;
    // With one motion and no clip of one's own to track, there is nothing to choose or say:
    // the result sheet already shows the dog that walk was traced from.
    element.hidden = single;
    if (single) {
      choice = { kind: "gallery", id };
      onChange();
    }
  }

  async function capture(file: File): Promise<void> {
    const current = ++selection;
    for (const radio of radios) {
      radio.checked = false;
    }
    choice = null;
    errors.replaceChildren();
    status.textContent = STRINGS.clipTracking;
    onChange();
    try {
      const job = await backend.capture(file);
      if (current !== selection) return;
      choice = { kind: "capture", job };
      status.textContent = "";
    } catch (error) {
      if (current !== selection) return;
      status.textContent = "";
      showError(errors, error);
    }
    onChange();
  }

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file !== undefined) {
      void capture(file);
    }
  });

  function showGallery(motions: readonly GalleryMotion[]): void {
    for (const motion of motions) {
      const radio = el("input", { type: "radio", name: "motion", value: motion.id });
      const thumbnail = el("img", { alt: "", class: "thumbnail" });
      radio.addEventListener("change", () => {
        selection += 1;
        choice = { kind: "gallery", id: motion.id };
        input.value = "";
        status.textContent = "";
        errors.replaceChildren();
        onChange();
      });
      radios.push(radio);
      gallery.append(
        el("label", { class: "motion-option" }, radio, thumbnail, STRINGS.motionName(motion.id)),
      );
      backend.file(motion.thumbnail).then(
        (blob) => {
          thumbnail.src = URL.createObjectURL(blob);
        },
        () => {
          // The label still names the motion when its thumbnail is missing.
          thumbnail.remove();
        },
      );
    }
    motionIds = motions.map((motion) => motion.id);
    settle();
  }

  function allowClips(allowed: boolean): void {
    clipLabel.hidden = !allowed;
    input.hidden = !allowed;
    clipsAllowed = allowed;
    settle();
  }

  return { element, choice: () => choice, showGallery, allowClips };
}

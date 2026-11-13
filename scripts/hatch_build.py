"""Build the app for source distributions and wheels, including wheels built from an sdist."""

import subprocess

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class AppBuildHook(BuildHookInterface[BuilderConfig]):
    """Keep release builds independent of a pre-existing static directory."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        """Generate the ignored assets before Hatch selects distribution files."""
        del build_data
        if version == "editable":
            return
        for command in (
            ["npm", "ci", "--prefix", "web"],
            ["npm", "run", "build", "--prefix", "web"],
        ):
            subprocess.run(command, cwd=self.root, check=True)  # noqa: S603

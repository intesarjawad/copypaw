"""User-facing errors: a code, its format arguments, and the text in messages.json."""

import json
from functools import cache
from importlib import resources
from typing import ClassVar, cast

__all__ = ["CopypawError", "InternalError", "message_table"]

_MESSAGES_NAME = "messages.json"


@cache
def message_table() -> dict[str, dict[str, str]]:
    """Return every code's `what` and `next` text, shared with the web build."""
    text = resources.files("copypaw").joinpath(_MESSAGES_NAME).read_text(encoding="utf-8")
    return cast("dict[str, dict[str, str]]", json.loads(text))


class CopypawError(Exception):
    """An error a person can act on, described by a code in messages.json.

    Subclasses list every code they may raise in `codes`; the first is the default.
    The format arguments fill the {placeholders} in that code's `what` and `next`.
    """

    codes: ClassVar[tuple[str, ...]] = ()
    exit_code: ClassVar[int] = 2

    def __init__(self, code: str | None = None, **arguments: object) -> None:
        """Select a code of this class and keep the values its text names."""
        if code is None:
            if not self.codes:
                raise TypeError(f"{type(self).__name__} needs a code")
            code = self.codes[0]
        elif code not in self.codes:
            raise ValueError(f"{type(self).__name__} cannot raise code {code!r}")
        self.code = code
        self.arguments = arguments
        super().__init__(self.what)

    @property
    def what(self) -> str:
        """One sentence naming what happened."""
        return message_table()[self.code]["what"].format(**self.arguments)

    @property
    def next(self) -> str:
        """The exact next command or control to use."""
        return message_table()[self.code]["next"].format(**self.arguments)


class InternalError(CopypawError):
    """Any exception Copypaw did not anticipate."""

    codes = ("internal",)
    exit_code = 1

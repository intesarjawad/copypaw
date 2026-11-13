"""Check public copy for contact details."""

import argparse
import re
import sys
from pathlib import Path

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def violations(text: str) -> list[tuple[int, str]]:
    """Return line numbers and descriptions of copy violations."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if EMAIL.search(line):
            found.append((number, "email address"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_dir():
                continue
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError as error:
                print(f"{file}: {error}", file=sys.stderr)
                failed = True
                continue
            for number, message in violations(content):
                print(f"{file}:{number}: {message}")
                failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())

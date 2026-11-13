# Contributing

Follow the [engineering standards](docs/engineering-standards.md) and
[code of conduct](CODE_OF_CONDUCT.md).

Use Python 3.12, 3.13, or 3.14 with uv 0.13.7. From the repository root:

```sh
make setup
make check
make build
```

## Contributor devcontainer

The `.devcontainer/` configuration builds a Python 3.12 container with Node 26 and
the OSMesa packages, sets `PYOPENGL_PLATFORM=osmesa`, and on creation runs
`uv sync`, `npm ci --prefix web`, and `copypaw models fetch --mode lightweight`. It
asks for a 2-core machine and forwards port 8765; start the app on it with
`uv run copypaw app --port 8765 --no-browser`.

Locally, with Docker running:

```sh
npx @devcontainers/cli up --workspace-folder .
npx @devcontainers/cli exec --workspace-folder . uv run pytest -m render
```

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/intesarjawad/copypaw)
The Codespaces badge needs a GitHub account.

`uv.lock` pins runtime and development dependencies. Update the lockfile when
changing a dependency. The `typings/` directory contains third-party declarations
only when a library's published types leave a gap.

The tracked pre-commit configuration runs ruff format, ruff check, pyright, and
gitleaks. Run the Python checks with `make check`. Contributors whose hooks are
managed externally should keep that configuration and run checks explicitly.

The files in `tests/golden/` pin the captured motion of the sample clip. They change
only by running `scripts/make_goldens.py`, and the pull request diff shows the
changed files for review.

Keep changes focused, add tests for behavior, and describe the reason for the
change and the checks you ran in the pull request. Do not include personal data,
credentials, model weights, or unlicensed assets. Report vulnerabilities through
the private channel described in [SECURITY.md](SECURITY.md).

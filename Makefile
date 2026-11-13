.PHONY: setup check check-web build

setup:
	uv sync --all-groups --locked
	npm ci --prefix web
	npm run build --prefix web

check: check-web
	uv run ruff format --check .
	uv run ruff check .
	uv run --with hatchling==1.33.0 pyright
	uv run pytest

check-web:
	npm run lint --prefix web
	npm run build --prefix web
	npm test --prefix web -- --maxWorkers=1 --testTimeout=120000 --retry=1

build:
	uv build

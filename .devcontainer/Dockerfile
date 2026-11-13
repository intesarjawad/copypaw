FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM mcr.microsoft.com/devcontainers/python:3.12@sha256:79a80515d78783e761af1db255e92767c4e887a2d56dbeb85e7753da8c12dc16

COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git make libosmesa6 libosmesa6-dev libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

ENV PYOPENGL_PLATFORM=osmesa

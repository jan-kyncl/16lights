# API i worker běží ze stejného image (liší se jen příkazem v docker-compose).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    L16_DATA_DIR=/data \
    L16_MODELS_DIR=/models \
    L16_LOG_JSON=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra demosaic --no-install-project

COPY l16core ./l16core
COPY app ./app
RUN uv sync --frozen --no-dev --extra demosaic

RUN useradd --uid 1000 --create-home l16 && mkdir -p /data /models && chown l16 /data /models
USER l16

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

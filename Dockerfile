FROM python:3.10-slim

# Optional install extras. Default (empty) = lean mock image. EXTRAS=lewm pulls torch +
# transformers for the real LeWorldModel backend (used by the docker-compose.lewm.yaml overlay).
ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir -e ".[$EXTRAS]"; else pip install --no-cache-dir -e .; fi

EXPOSE 8080
CMD ["uvicorn", "wmcp_jepa_service.server:app", "--host", "0.0.0.0", "--port", "8080"]

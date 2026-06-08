# Push-T WMCP demo client — stdlib only, so no pip install is needed for the default demo.
FROM python:3.10-slim

WORKDIR /app
COPY client ./client

# Service address inside docker compose; override with WMCP_BASE_URL.
ENV WMCP_BASE_URL=http://wmcp-jepa-service:8080 \
    WMCP_MODEL_ID=lewm-pusht \
    PYTHONUNBUFFERED=1

# Runs one metadata -> score -> plan cycle and prints a human-readable result.
CMD ["python", "-m", "client.demo"]

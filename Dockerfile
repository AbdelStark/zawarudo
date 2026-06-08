FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["uvicorn", "wmcp_jepa_service.server:app", "--host", "0.0.0.0", "--port", "8080"]

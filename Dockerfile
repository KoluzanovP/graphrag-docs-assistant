# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps: build tools for psycopg2 / sentence-transformers wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Serve the FastAPI app (which also serves the static frontend).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile for HF Spaces — Option A: FastAPI + Streamlit combined
#
# ARCHITECTURE:
#   Port 7860 (HF public) → FastAPI (uvicorn)
#     ├── /whatsapp/webhook  → Twilio WhatsApp handler
#     ├── /health            → health check
#     └── /*                 → reverse proxy → Streamlit on port 8501
#
# Streamlit runs internally on 8501, started as a subprocess by hf_app.py.
# Only port 7860 is exposed to the internet.

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first (fixes "requirements.txt not found" in Docker build)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data dir for SQLite
RUN mkdir -p /app/data

# HF runs containers as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HF Docker Spaces require port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# hf_app.py starts Streamlit on 8501 then FastAPI on 7860
CMD ["python", "hf_app.py"]
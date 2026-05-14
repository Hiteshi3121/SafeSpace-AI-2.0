# ── Dockerfile for Hugging Face Spaces (Docker SDK) ──────────────────────────
#
# HF Spaces removed 'streamlit' as a create_repo SDK option from its API.
# The modern way to run Streamlit on HF is the Docker SDK — you provide
# this Dockerfile and HF builds + runs it.
#
# HF Docker Spaces requirements:
#   - App must listen on port 7860 (HF routes traffic there)
#   - Container runs as non-root user (uid 1000)
#   - No persistent storage — data/ is ephemeral per session
#
# HOW HF USES THIS:
#   HF detects Dockerfile → builds image → runs it
#   Our CMD starts Streamlit on port 7860

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data dir for SQLite (ephemeral on HF — resets on restart)
RUN mkdir -p /app/data

# HF runs containers as uid 1000 — make data dir writable
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Port 7860 is required by HF Docker Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Start Streamlit via hf_app.py (which sets up sys.path and env vars)
CMD ["streamlit", "run", "hf_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none", \
     "--browser.gatherUsageStats", "false"]
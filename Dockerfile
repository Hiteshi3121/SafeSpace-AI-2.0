# Dockerfile for Hugging Face Spaces (Docker SDK)
# App runs on port 7860 (required by HF)

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL files first, then install
# (avoids "requirements.txt not found" if COPY order is wrong)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data dir for SQLite
RUN mkdir -p /app/data

# HF runs as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Port 7860 required by HF Docker Spaces
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "hf_app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none", \
     "--browser.gatherUsageStats", "false"]
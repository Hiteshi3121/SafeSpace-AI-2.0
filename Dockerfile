# Dockerfile — SafeSpace AI (FastAPI + Streamlit + Nginx)
#
# ARCHITECTURE:
#   Nginx on port 7860 (HF public port) routes by URL path:
#     /whatsapp/*  → FastAPI on 8000 (Twilio webhook)
#     /health      → FastAPI on 8000
#     /*           → Streamlit on 8501 (Web UI)
#
# This solves the WebSocket 403 problem because Nginx properly
# proxies WebSocket connections unlike FastAPI's reverse proxy.

FROM python:3.11-slim

WORKDIR /app

# Install system deps including nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data dir
RUN mkdir -p /app/data

# Write nginx config
RUN cat > /etc/nginx/nginx.conf << 'NGINXEOF'
worker_processes 1;
events { worker_connections 1024; }

http {
    # Route /whatsapp and /health to FastAPI
    # Route everything else (including WebSocket) to Streamlit
    
    server {
        listen 7860;
        
        # WhatsApp webhook → FastAPI
        location /whatsapp/ {
            proxy_pass http://127.0.0.1:8000/whatsapp/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        # Health check → FastAPI
        location /health {
            proxy_pass http://127.0.0.1:8000/health;
            proxy_set_header Host $host;
        }
        
        # Everything else → Streamlit (including WebSocket)
        location / {
            proxy_pass http://127.0.0.1:8501/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # Critical: proper WebSocket proxying
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }
    }
}
NGINXEOF

# HF runs as uid 1000
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /var/log/nginx && \
    chown -R appuser:appuser /var/lib/nginx && \
    chown -R appuser:appuser /etc/nginx && \
    mkdir -p /tmp/nginx && chown appuser:appuser /tmp/nginx

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "hf_app.py"]
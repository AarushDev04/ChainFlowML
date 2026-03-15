# ==============================================================================
# CHAINFLOW AI — docker/Dockerfile
# ==============================================================================

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS production
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app/       ./app/
COPY scripts/   ./scripts/
RUN mkdir -p /app/artifacts/models /app/artifacts/chromadb \
             /app/artifacts/results /app/data /tmp/chainflow_memory
RUN useradd -m -u 1000 chainflow && chown -R chainflow /app /tmp/chainflow_memory
USER chainflow
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1","--log-level","info"]

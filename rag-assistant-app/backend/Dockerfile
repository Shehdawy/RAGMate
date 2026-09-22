# Build from the repository root:  docker build -f backend/Dockerfile -t documind-backend .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so the container starts without internet access to the model hub
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend/app ./app
COPY backend/data ./data
COPY data/raw ./data/raw

# Run as a non-root user; the vector store directory must be writable (uploads, first-start indexing)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/vector_store \
    && chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# --proxy-headers: trust X-Forwarded-* from the reverse proxy in front of the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

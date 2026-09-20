FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container starts offline-ready
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY data ./data

EXPOSE 8000
# Ollama runs on the host: set OLLAMA_HOST=http://host.docker.internal:11434
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

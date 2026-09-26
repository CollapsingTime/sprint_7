FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAG_INDEX_DIR=/app/index \
    RAG_MODEL_CACHE=/app/models \
    RAG_LLM_BASE_URL=http://localhost:11434/v1 \
    RAG_LLM_MODEL=qwen2.5:7b \
    RAG_TOP_K=3 \
    RAG_MIN_SCORE=0.80

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.14.0+cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY rag_bot.py ./

EXPOSE 8000

CMD ["uvicorn", "rag_bot:app", "--host", "0.0.0.0", "--port", "8000"]

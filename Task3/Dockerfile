FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.14.0+cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY build_index.py test_index.py ./

CMD ["python", "build_index.py"]

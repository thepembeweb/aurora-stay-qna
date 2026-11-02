
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/huggingface
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='HarishMaths/Hotel-Policy-Embedding', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})"
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_service.py rag.py store_in_faiss.py index.html ./
COPY chunked_data.json faiss_index.bin ./

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]

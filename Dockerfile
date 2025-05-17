FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Configure environment variables
ENV MODEL_NAME="BAAI/bge-m3"
ENV MODEL_DEVICE="cpu"
ENV USE_FP16="false"
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Use gunicorn as the production-ready WSGI server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

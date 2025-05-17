# BGE-M3 Embedding API Service

This project containerizes the BGE-M3 embedding model, deploys it to Google Cloud, and provides a secure API for generating embeddings.

## Features

- Flask API for BGE-M3 model
- Containerized with Docker
- Google Cloud Run deployment
- API key authentication
- Support for dense and sparse embeddings

## Local Development

### Prerequisites

- Python 3.8+
- Docker

### Setup and Testing

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the local development server:

```bash
python app.py
```

3. After verifying that the app works locally, stop the running app (Ctrl+C) and build the Docker image:

```bash
# Navigate to the directory containing the Dockerfile
cd /path/to/bge_m3_api

# Build the Docker image with a tag
docker build -t bge-m3-service .
```

4. Run the container:

```bash
docker run -p 8080:8080 -e "ALLOWED_API_KEYS=test_key" bge-m3-service
```

5. Test the containerized API:

```bash
# Test the health endpoint
curl http://localhost:8080/health

# Test the encode endpoint (requires API key)
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_key" \
  -d '{"texts": ["This is a test sentence."]}'
```

## Google Cloud Deployment

1. Make the deployment script executable:

```bash
chmod +x gcp_deploy.sh
```

2. Edit the script to set your GCP project ID and other configuration.

3. Run the deployment:

```bash
./gcp_deploy.sh
```

## API Endpoints

### Health Check


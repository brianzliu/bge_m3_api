#!/bin/bash

# Configuration - CHANGE THESE VALUES
PROJECT_ID="gen-lang-client-0314566346"
IMAGE_NAME="bge-m3-service"
REGION="us-central1"
SERVICE_NAME="bge-m3-api"

# Build the Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
gcloud auth configure-docker
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars="ALLOWED_API_KEYS=key1,key2,key3" \
  --no-allow-unauthenticated

echo "Deployment completed! Service URL:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'

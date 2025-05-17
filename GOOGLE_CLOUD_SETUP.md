# Google Cloud Setup for BGE-M3 API Service

This guide walks you through setting up your Google Cloud environment and finding all the parameters needed for the `gcp_deploy.sh` script.

## Prerequisites

1. [Google Cloud account](https://cloud.google.com/)
2. [Google Cloud CLI installed](https://cloud.google.com/sdk/docs/install)
3. Docker installed locally

## Step 1: Set Up Google Cloud Project

### Create a New Project (or use an existing one)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Enter a project name and click "Create"

### Enable Required APIs

1. Go to [API Library](https://console.cloud.google.com/apis/library)
2. Search for and enable these APIs:
   - Cloud Run API
   - Container Registry API
   - Cloud Build API
   - Artifact Registry API (if using Artifact Registry instead of Container Registry)

## Step 2: Find Parameters for gcp_deploy.sh

Here's where to find each parameter needed in the deployment script:

### PROJECT_ID

This is your Google Cloud project ID.

1. Go to the [Dashboard](https://console.cloud.google.com/home/dashboard)
2. Find your project ID at the top of the page or in the project info card

Example: `my-bge-embeddings-project`

### IMAGE_NAME

This is a name you choose for your Docker image.

Example: `bge-m3-service`

### REGION

Choose a Google Cloud region where you want to deploy your service.

1. Go to [Cloud Run](https://console.cloud.google.com/run)
2. Look at the "Region" dropdown for available options

Common regions:
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `us-west1` (Oregon)
- `europe-west1` (Belgium)
- `asia-east1` (Taiwan)

Choose a region that's close to your users for lower latency.

### SERVICE_NAME

This is a name you choose for your Cloud Run service.

Example: `bge-m3-api`

## Step 3: Configure Authentication

### Set up API Keys

Decide what API keys you want to allow and add them to the `ALLOWED_API_KEYS` environment variable in the script.

Example: `--set-env-vars="ALLOWED_API_KEYS=key1,key2,key3"`

### Authenticate with Google Cloud

Before running the deploy script:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Step 4: Customize Memory and CPU Settings

Based on BGE-M3 model requirements, you might want to adjust:

```bash
--memory 4Gi \  # Increase to 8Gi for better performance
--cpu 2 \       # Increase to 4 for better performance
```

For GPU acceleration (optional, higher cost):
```bash
--gpu 1 \
--gpu-type=nvidia-tesla-t4
```

Note: Not all regions support GPU on Cloud Run, and it significantly increases cost.

## Step 5: Update gcp_deploy.sh

Edit your `gcp_deploy.sh` script with all the parameters you've gathered:

```bash
#!/bin/bash

# Configuration - CHANGE THESE VALUES
PROJECT_ID="your-project-id-here"
IMAGE_NAME="bge-m3-service"
REGION="us-central1"
SERVICE_NAME="bge-m3-api"

# ... rest of script ...
```

## Step 6: Run the Deployment

```bash
chmod +x gcp_deploy.sh
./gcp_deploy.sh
```

## Monitoring Your Deployment

After deployment:

1. Go to [Cloud Run](https://console.cloud.google.com/run)
2. Click on your service name
3. Monitor logs, traffic, and metrics

## Cost Management

- Cloud Run charges based on usage (requests, CPU, memory)
- Enable [budget alerts](https://console.cloud.google.com/billing/budgets) to avoid unexpected charges
- Consider setting concurrency and maximum instances to control costs:
  ```
  --concurrency=80 --max-instances=10
  ```

## Alternative Deployment Options

### Vertex AI (for Production ML Workloads)

For production use with higher throughput requirements, consider:

1. [Vertex AI Custom Model deployment](https://console.cloud.google.com/vertex-ai/model-deployment)
2. Supports Docker containers and has built-in scaling

### Google Kubernetes Engine (GKE)

For complex deployments with multiple services:

1. [GKE](https://console.cloud.google.com/kubernetes) provides more control
2. Better for production workloads that need custom networking, scaling policies, etc.

import requests
import json
import time
import argparse

def encode_texts(api_url, api_key, texts, params=None):
    """
    Send texts to the BGE-M3 API for encoding.
    
    Args:
        api_url (str): The URL of the API endpoint
        api_key (str): API key for authentication
        texts (list): List of texts to encode
        params (dict): Additional parameters for encoding
        
    Returns:
        dict: The embeddings response from the API
    """
    if params is None:
        params = {}
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    payload = {
        "texts": texts,
        "return_dense": params.get("return_dense", True),
        "return_sparse": params.get("return_sparse", False),
        "return_colbert_vecs": params.get("return_colbert_vecs", False),
        "batch_size": params.get("batch_size", 32),
        "max_length": params.get("max_length", 8192)
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Response: {response.text}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_health(api_url):
    """Check if the API is healthy and the model is loaded"""
    try:
        response = requests.get(f"{api_url}/health")
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def wait_for_model_loading(api_url, max_wait_seconds=300, check_interval=5):
    """Wait until the model is loaded and ready to use"""
    print("Checking if model is loaded...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        health = check_health(api_url)
        
        if health.get("model_loaded", False):
            print("Model is loaded and ready!")
            return True
            
        print(f"Model is still loading. Waiting {check_interval} seconds...")
        time.sleep(check_interval)
    
    print(f"Timed out after {max_wait_seconds} seconds. Model might still be loading.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 API Client Example")
    parser.add_argument("--api-url", type=str, required=True, help="API URL (e.g. https://bge-m3-api-xxx.run.app)")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    
    args = parser.parse_args()
    
    # Full API URL for the encode endpoint
    encode_url = f"{args.api_url}/encode"
    
    # Wait for the model to be ready
    if not wait_for_model_loading(args.api_url):
        print("Proceeding anyway, but requests might fail if model isn't ready.")
    
    # Example texts to encode
    example_texts = [
        "This is a sample document for embedding generation.",
        "BGE-M3 is a multi-modal embedding model.",
        "Cloud deployment allows for scalable API access."
    ]
    
    # Encode the texts
    print("\nEncoding texts...")
    result = encode_texts(encode_url, args.api_key, example_texts)
    
    if result:
        print("\nEncoding successful!")
        
        # Display the dense vectors dimensions (just the first few values for the first text)
        if 'dense_vecs' in result:
            first_vector = result['dense_vecs'][0]
            print(f"\nDense vector dimension: {len(first_vector)}")
            print(f"First few values: {first_vector[:5]}")
        
        # Display any other returned data
        for key in result:
            if key != 'dense_vecs':
                print(f"\nReturned {key} data")

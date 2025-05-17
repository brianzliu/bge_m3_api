from flask import Flask, request, jsonify
from flask_cors import CORS
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import os
import logging
import time
from functools import wraps
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables
MODEL = None
MODEL_LOADED = threading.Event()

# Initialize API_KEYS
_raw_api_keys_env = os.environ.get('ALLOWED_API_KEYS')
if _raw_api_keys_env:
    # Filter out empty strings that can result from splitting an empty string
    # or strings with just commas, and strip whitespace from keys.
    API_KEYS = set(key.strip() for key in _raw_api_keys_env.split(',') if key.strip())
else:
    # If ALLOWED_API_KEYS is not set or is an empty string,
    # API_KEYS will be an empty set.
    # The decorator `require_api_key` will treat an empty API_KEYS set as "no auth required".
    API_KEYS = set()

logger.info(f"API Keys loaded: {API_KEYS if API_KEYS else 'No API keys configured (open access)'}")

def initialize_bge_m3_model(model_name='BAAI/bge-m3', use_fp16=None, device=None):
    """
    Initialize the BGE-M3 model and signal when it's ready to use.
    
    Args:
        model_name (str): The model identifier to load
        use_fp16 (bool): Whether to use FP16 precision (defaults to True if on GPU, False if on CPU)
        device (str): Device to use ('cpu', 'cuda:0', etc.)
        
    Returns:
        BGEM3FlagModel: The initialized model
    """
    global MODEL
    
    # Determine device if not specified
    if device is None:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Set default use_fp16 based on device if not specified
    if use_fp16 is None:
        use_fp16 = device.startswith('cuda')
    
    logger.info(f"Initializing BGE-M3 model on device: {device} with FP16: {use_fp16}")
    start_time = time.time()
    
    try:
        MODEL = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        load_time = time.time() - start_time
        logger.info(f"Model initialization complete in {load_time:.2f} seconds")
        MODEL_LOADED.set()  # Signal that the model is ready
        return MODEL
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def require_api_key(f):
    """Decorator to require a valid API key for endpoint access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If API_KEYS is empty (meaning no keys were configured) or contains '*', allow access.
        if not API_KEYS or '*' in API_KEYS:
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
            logger.warning(f"Access denied: Invalid or missing API key. Provided API key: '{api_key}'. Allowed keys: {API_KEYS}")
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

def wait_for_model(f):
    """Decorator to ensure model is loaded before processing requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not MODEL_LOADED.is_set():
            return jsonify({"error": "Model is still loading. Please try again later."}), 503
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is healthy and model is loaded"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": MODEL_LOADED.is_set()
    }), 200 if MODEL_LOADED.is_set() else 503

@app.route('/encode', methods=['POST'])
@require_api_key
@wait_for_model
def encode_texts():
    """Endpoint to encode texts with the BGE-M3 model"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' in request body"}), 400
        
        texts = data['texts']
        
        # Get parameters with defaults
        params = {
            'return_dense': data.get('return_dense', True),
            'return_sparse': data.get('return_sparse', False),
            'return_colbert_vecs': data.get('return_colbert_vecs', False),
            'batch_size': data.get('batch_size', 32),
            'max_length': data.get('max_length', 8192)
        }
        
        # Get embeddings from the model
        embeddings = MODEL.encode(texts, **params)
        
        # Convert numpy arrays to Python lists for JSON serialization
        result = {}
        if 'dense_vecs' in embeddings and embeddings['dense_vecs'] is not None:
            result['dense_vecs'] = embeddings['dense_vecs'].tolist()
        
        if 'lexical_weights' in embeddings and embeddings['lexical_weights'] is not None:
            result['lexical_weights'] = embeddings['lexical_weights']
            
        if 'colbert_vecs' in embeddings and embeddings['colbert_vecs'] is not None:
            result['colbert_vecs'] = [vec.tolist() for vec in embeddings['colbert_vecs']]
        
        # Compute and add pairwise Colbert scores if requested
        if data.get('compute_colbert_pairwise_scores', False) and \
           params['return_colbert_vecs'] and \
           'colbert_vecs' in result and \
           len(texts) > 1:
            
            # The 'colbert_vecs' in embeddings are already in the correct format
            # (list of individual text Colbert vectors) needed by MODEL.colbert_score
            colbert_vectors_for_scoring = embeddings['colbert_vecs']
            num_vectors = len(colbert_vectors_for_scoring)
            pairwise_scores = []
            for i in range(num_vectors):
                for j in range(i + 1, num_vectors):
                    score = MODEL.colbert_score(colbert_vectors_for_scoring[i], colbert_vectors_for_scoring[j])
                    # Ensure score is a Python float for JSON serialization
                    if hasattr(score, 'item'): # Check if it's a PyTorch tensor
                        score = score.item()
                    pairwise_scores.append({
                        "text_indices": [i, j],
                        "score": score
                    })
            result['colbert_pairwise_scores'] = pairwise_scores
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/colbert-similarity', methods=['POST'])
@require_api_key
@wait_for_model
def colbert_similarity_to_query():
    """
    Computes ColBERT-style similarity scores between a query text and multiple candidate texts.
    Expects JSON: {"query_text": "...", "candidate_texts": ["...", "..."]}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        query_text = data.get('query_text')
        candidate_texts = data.get('candidate_texts')

        if not query_text or not isinstance(query_text, str):
            return jsonify({"error": "Missing or invalid 'query_text'"}), 400
        if not candidate_texts or not isinstance(candidate_texts, list) or not all(isinstance(t, str) for t in candidate_texts):
            return jsonify({"error": "Missing or invalid 'candidate_texts'"}), 400
        if not candidate_texts:
            return jsonify({"scores": []}), 200 # No candidates, return empty scores

        all_texts = [query_text] + candidate_texts
        
        # Encode all texts to get Colbert vectors
        # Only need colbert_vecs, so set others to False for efficiency
        embeddings_output = MODEL.encode(
            all_texts, 
            return_dense=False, 
            return_sparse=False, 
            return_colbert_vecs=True
        )
        
        colbert_vecs = embeddings_output.get('colbert_vecs')
        if not colbert_vecs or len(colbert_vecs) != len(all_texts):
            logger.error("Failed to get valid Colbert vectors from model encoding.")
            return jsonify({"error": "Failed to compute Colbert vectors"}), 500

        query_colbert_vec = colbert_vecs[0]
        candidate_colbert_vecs = colbert_vecs[1:]
        
        scores = []
        for cand_vec in candidate_colbert_vecs:
            score = MODEL.colbert_score(query_colbert_vec, cand_vec)
            if hasattr(score, 'item'): # Handle PyTorch tensor
                score = score.item()
            scores.append(score)
            
        return jsonify({"scores": scores}), 200

    except Exception as e:
        logger.error(f"Error in /colbert-similarity: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500

# Start model initialization in a separate thread
def start_model_initialization():
    """Start model initialization in a background thread"""
    def initialize_model_thread():
        try:
            model_name = os.environ.get('MODEL_NAME', 'BAAI/bge-m3')
            use_fp16_str = os.environ.get('USE_FP16', '').lower()
            use_fp16 = None
            if use_fp16_str in ('true', 'false'):
                use_fp16 = use_fp16_str == 'true'
            device = os.environ.get('MODEL_DEVICE', None)
            
            initialize_bge_m3_model(model_name=model_name, use_fp16=use_fp16, device=device)
        except Exception as e:
            logger.error(f"Background model initialization failed: {str(e)}")
            
    thread = threading.Thread(target=initialize_model_thread)
    thread.daemon = True
    thread.start()

start_model_initialization() # Ensure model initialization starts when module is loaded

if __name__ == '__main__':
    # Start model initialization in background
    # start_model_initialization() # No longer needed here as it's called at module level
    
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

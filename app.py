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
API_KEYS = set(os.environ.get('ALLOWED_API_KEYS', '').split(','))

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
        if not API_KEYS or '*' in API_KEYS:  # Skip validation if no keys defined or wildcard is set
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
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
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    # Start model initialization in background
    start_model_initialization()
    
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from werkzeug.exceptions import BadRequest
import os
import threading

app = Flask(__name__)

# CONNECT TO Hugging Face
# We get the token from Vercel's "Environment Variables" for security
# Client initialization is deferred to route handler to avoid import-time errors
_client = None
_client_lock = threading.Lock()

def get_client():
    global _client
    if _client is None:
        with _client_lock:
            # Double-check after acquiring lock
            if _client is None:
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    return None
                _client = InferenceClient(token=hf_token)
    return _client

@app.route('/', methods=['POST'])
def chat():
    # Check for HF_TOKEN configuration
    client = get_client()
    if client is None:
        return jsonify({"error": "Server configuration error: HF_TOKEN not set"}), 500
    
    # Validate content type and parse JSON
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        data = request.get_json()
    except BadRequest:
        return jsonify({"error": "Invalid JSON in request body"}), 400
    
    # Validate that data is a dict and message is a string
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    
    message = data.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "Message must be a string"}), 400
    
    user_message = message.strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Send to Hugging Face
        response = client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=500
        )
        
        # Get the answer
        ai_reply = response.choices[0].message.content
        return jsonify({"reply": ai_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# CONNECT TO HUGGING FACE
# We get the token from Vercel's "Environment Variables" for security
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required")

client = InferenceClient(token=hf_token)

@app.route('/', methods=['POST'])
def chat():
    # Validate content type and parse JSON
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON in request body"}), 400
    
    user_message = data.get("message", "")

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

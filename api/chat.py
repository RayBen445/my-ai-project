from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# CONNECT TO HUGGING FACE
# We get the token from Vercel's "Environment Variables" for security
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

@app.route('/', methods=['POST'])
def handler():
    data = request.json
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

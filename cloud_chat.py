from huggingface_hub import InferenceClient

# 1. SETUP: Connect to the Cloud
# Paste your token inside the quotes below
my_token = ""

# We use a powerful model hosted by Hugging Face (Free Tier)
# You can change this to "meta-llama/Meta-Llama-3-8B-Instruct" if you want
repo_id = "Qwen/Qwen2.5-72B-Instruct"

print(f"☁️ Connecting to Hugging Face Cloud ({repo_id})...")
client = InferenceClient(token=my_token)

# 2. THE CHAT LOOP
history = []

print("\n✅ CONNECTED. I am running on the cloud, not your phone.")
print("Type 'exit' to stop.\n")

while True:
    try:
        user_input = input("\n👤 Professor: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Disconnecting...")
            break

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        print("🤖 AI: ", end="", flush=True)

        # Send data to Hugging Face and stream the response back
        stream = client.chat_completion(
            model=repo_id,
            messages=history,
            max_tokens=500,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            # Extract the text chunks from the cloud response
            if chunk.choices and chunk.choices[0].delta.content:
                text_part = chunk.choices[0].delta.content
                print(text_part, end="", flush=True)
                full_response += text_part
        
        print() # New line
        
        # Remember the AI's answer
        history.append({"role": "assistant", "content": full_response})

    except Exception as e:
        print(f"\n❌ Cloud Error: {e}")
        print("Note: If the model is busy, try again in a few seconds.")


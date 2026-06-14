import os
import sys
from pathlib import Path

# Add the parent 'src' directory to sys.path to allow sibling module imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fixtures import OPENAI_API_KEY, MODEL_NAME
from openai import OpenAI

def main():
    
    # Retrieve the API key from environment
    api_key = OPENAI_API_KEY
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        print("Please set it in your shell environment or add it to the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    print("Initializing OpenAI client...")
    client = OpenAI(api_key=api_key)
    
    try:
        print("Sending a test chat completion request to OpenAI...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Hello from OpenAI! Connection is successful.'"}
            ]
        )
        
        print("\n--- OpenAI Response ---")
        print(response.choices[0].message.content)
        print("-----------------------")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
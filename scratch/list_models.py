import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("No GOOGLE_API_KEY found.")
    sys.exit(1)

import google.generativeai as genai

genai.configure(api_key=google_api_key)

try:
    print("Available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- Name: {m.name}")
            print(f"  DisplayName: {m.display_name}")
            print(f"  Description: {m.description}")
            print(f"  InputTokenLimit: {m.input_token_limit}")
            print(f"  OutputTokenLimit: {m.output_token_limit}")
except Exception as e:
    print(f"Error listing models: {e}")

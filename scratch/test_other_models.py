import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("No GOOGLE_API_KEY found.")
    sys.exit(1)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

models_to_test = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3.5-flash",
    "gemini-1.5-pro",
]

for model in models_to_test:
    try:
        print(f"Testing model: {model}...")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0.0,
            max_output_tokens=10,
        )
        res = llm.invoke([HumanMessage(content="Say Hello.")])
        print(f"-> Success: {res.content.strip()}")
    except Exception as e:
        print(f"-> Failed for {model}: {e}")

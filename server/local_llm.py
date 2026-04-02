# local_llm.py
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = None

def get_client():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        client = Groq(api_key=api_key)
    return client

def get_completion(prompt: str, temperature=0.7, max_tokens=1024):
    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",   # Change to "llama-3.3-70b-versatile" when ready
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
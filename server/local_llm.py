# local_llm.py
"""
Groq LLM client wrapper for both agents
"""

import os
from groq import Groq
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

client = None

def get_groq_client():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        client = Groq(api_key=api_key)
    return client


def get_llm_response(prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
    """Get response from Groq LLM"""
    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.1-8b-instant",   # Good balance of speed and quality
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return "Failed to get LLM response."

# Add this to your local_llm.py
def get_llm_response_with_history(system_prompt: str, user_messages: list, temperature=0.7):
    """Send conversation with system prompt"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_messages)
    
    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=temperature,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "Error generating response."
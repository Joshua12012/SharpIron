
"""
Clean Groq LLM wrapper - using fast model
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = None
MODEL_NAME = os.getenv("GROQ_MODEL")
def get_groq_client():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] No LLM provider configured! Set either HF_TOKEN or GROQ_API_KEY", flush=True)
            return None
        client = Groq(api_key=api_key)
    return client


def get_llm_response(messages: list, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Main function used by agents - fast and safe"""
    # Ensure all content is string
    safe_messages = []
    for msg in messages:
        safe_msg = dict(msg)  # copy
        if not isinstance(safe_msg.get("content"), str):
            safe_msg["content"] = str(safe_msg.get("content", ""))
        safe_messages.append(safe_msg)

    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=safe_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return "Failed to get LLM response."
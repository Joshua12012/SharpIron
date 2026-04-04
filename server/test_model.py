import local_llm
messages = [{'role': 'user', 'content': 'Hello'}]
print("Calling model:", "llama-3.1-8b-instant")
print("Response:", local_llm.get_llm_response(messages))

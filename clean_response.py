# clean_response.py

def clean_response(response: str) -> str:
    raw_text = str(response).strip()
    prefixes = ["Đáp assistant", "assistant:", "Assistant:", "Đáp:", "Đáp"]
    for prefix in prefixes:
        if raw_text.lower().startswith(prefix.lower()):
            return raw_text[len(prefix):].strip()
    return raw_text
from groq import Groq
from modules.config import get_secret

def _client():
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your environment.")
    return Groq(api_key=key)

def answer_text(prompt: str, history=None) -> str:
    client = _client()
    messages = [{
        "role": "system",
        "content": "You are a helpful AI assistant. Be accurate, clear, and acknowledge uncertainty."
    }]
    for item in (history or [])[-10:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    model = get_secret("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content or "The model returned an empty response."

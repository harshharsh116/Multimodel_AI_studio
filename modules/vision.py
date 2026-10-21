import base64
import io
from groq import Groq
from modules.config import get_secret

def inspect_image(image, question: str) -> str:
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your environment.")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    model = get_secret("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content or "The model returned an empty response."

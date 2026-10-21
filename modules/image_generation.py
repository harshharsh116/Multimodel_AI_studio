import io
from huggingface_hub import InferenceClient
from modules.config import get_secret

def _client():
    token = get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add a Hugging Face access token to .streamlit/secrets.toml.")
    return InferenceClient(token=token)

def generate_image(prompt: str):
    model = get_secret("HF_TEXT_TO_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    image = _client().text_to_image(prompt, model=model)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def edit_image(image, prompt: str):
    model = get_secret("HF_IMAGE_TO_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
    result = _client().image_to_image(image=image, prompt=prompt, model=model)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

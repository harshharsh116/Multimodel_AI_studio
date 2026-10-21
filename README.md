# Multimodal AI Studio

A Streamlit prototype combining hosted text/vision inference, hosted image generation/editing, and local semantic retrieval for PDF RAG.

## Features
- AI chat with short session history
- Image understanding / visual question answering
- Text-to-image generation
- Image-to-image editing
- PDF ingestion, local Sentence Transformers embeddings, FAISS retrieval, and grounded answers with source passages

## Requirements
- Python 3.11 recommended
- Groq API key for chat, visual question answering, and RAG answer generation
- Hugging Face access token for image generation/editing
- Internet access for API calls and the first download of the local embedding model

Model/provider access and pricing can change. Check current provider model availability before deployment.

## Local setup (Windows / PyCharm)
```powershell
cd path\to\multimodal_ai_studio
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir .streamlit
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` and add your real keys. Never commit this file.

Run:
```powershell
streamlit run app.py
```

## API keys
- Groq: create an API key in your Groq account dashboard.
- Hugging Face: create a token with the permissions required for inference.

Never paste keys into source code, commit them, or expose them in screenshots.

## Deployment
1. Push the project to a GitHub repository without `.streamlit/secrets.toml`, `.env`, model weights, or private documents.
2. Create a Streamlit Community Cloud app using `app.py` as the entrypoint.
3. Add the same TOML key/value entries from your local secrets file in the app's Secrets settings.
4. Deploy and check logs if a provider/model is unavailable.
5. If dependency installation fails due to memory limits, split heavy local embedding work into a separate service or use a lighter embedding deployment.

## Limitations of this starter
- Scanned PDFs without selectable text need OCR; not included yet.
- PDF indexing is held in Streamlit session state, so it is rebuilt after a session reset.
- Image editing/generation endpoints and model access depend on current provider availability and account permissions.
- Add authentication, rate limiting, quotas, persistent storage, and file-size controls before public production use.
- Do not upload sensitive documents unless you understand the provider's data handling and retention terms.

import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer
from modules.text_ai import answer_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _embedder():
    # Downloads the embedding model on first use; embeddings run locally.
    return SentenceTransformer(EMBEDDING_MODEL)

def _split_text(text: str, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def index_pdfs(uploaded_files):
    chunks, metadata = [], []
    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        document = fitz.open(stream=data, filetype="pdf")
        try:
            for page_num, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                for chunk in _split_text(page_text):
                    chunks.append(chunk)
                    metadata.append({
                        "filename": uploaded.name,
                        "page": page_num,
                        "text": chunk,
                    })
        finally:
            document.close()

    if not chunks:
        raise ValueError("No selectable text was found. This version does not yet OCR scanned PDFs.")

    model = _embedder()
    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return chunks, index, metadata

def answer_from_pdfs(question, chunks, index, metadata, top_k=4):
    if not chunks or index.ntotal == 0:
        raise ValueError("No indexed document text is available. Upload and index PDFs first.")
    model = _embedder()
    query_vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, ids = index.search(query_vector, min(top_k, index.ntotal))

    retrieved = []
    for idx, score in zip(ids[0], scores[0]):
        if idx >= 0:
            item = metadata[int(idx)]
            retrieved.append({**item, "score": float(score)})

    context = "\n\n".join(
        f"[Source {i+1}: {item['filename']}, page {item['page']}]\n{item['text']}"
        for i, item in enumerate(retrieved)
    )
    prompt = f"""Answer the user's question using ONLY the context below.
If the context does not contain the answer, say that the uploaded documents do not provide enough information.
Do not invent facts. Refer to source filenames and page numbers where useful.

CONTEXT:
{context}

QUESTION:
{question}
"""
    answer = answer_text(prompt)
    return answer, retrieved

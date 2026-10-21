import streamlit as st
from PIL import Image
from modules.text_ai import answer_text
from modules.vision import inspect_image
from modules.image_generation import generate_image, edit_image
from modules.rag import index_pdfs, answer_from_pdfs

st.set_page_config(page_title="Multimodal AI Studio", page_icon="✨", layout="wide")
st.title("✨ Multimodal AI Studio")
st.caption("One workspace for chat, image understanding, image generation, and PDF question answering.")

with st.sidebar:
    st.header("Choose a tool")
    mode = st.radio(
        "AI tools",
        ["AI Chat", "Image to Text / Visual QA", "Text to Image", "Image to Image", "Chat with PDF (RAG)"],
    )
    st.divider()
    st.caption("Hybrid architecture: hosted inference for generation/chat; local embeddings for RAG.")

def show_error(exc):
    st.error(str(exc))
    st.info("Check the API keys in .streamlit/secrets.toml, provider/model availability, and your internet connection.")

if mode == "AI Chat":
    st.subheader("AI Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            with st.spinner("Thinking..."):
                reply = answer_text(prompt, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        except Exception as e:
            show_error(e)

elif mode == "Image to Text / Visual QA":
    st.subheader("Image understanding")
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key="vision_upload")
    question = st.text_input("What should I do with the image?", "Describe this image in detail.")
    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze image", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    result = inspect_image(image, question)
                st.markdown("### Result")
                st.write(result)
            except Exception as e:
                show_error(e)

elif mode == "Text to Image":
    st.subheader("Generate an image from text")
    prompt = st.text_area("Describe the image you want", placeholder="A cinematic photograph of a futuristic city in Gujarat at sunset...")
    if st.button("Generate image", type="primary", disabled=not prompt.strip()):
        try:
            with st.spinner("Generating image. This can take a little while..."):
                result = generate_image(prompt)
            st.image(result, caption="Generated image", use_container_width=True)
            st.download_button("Download image", result.getvalue(), file_name="generated_image.png", mime="image/png")
        except Exception as e:
            show_error(e)

elif mode == "Image to Image":
    st.subheader("Transform an existing image")
    upload = st.file_uploader("Upload a source image", type=["png", "jpg", "jpeg", "webp"], key="edit_upload")
    prompt = st.text_area("Describe the transformation", placeholder="Turn this into a watercolor illustration while keeping the main subject...")
    if upload:
        source = Image.open(upload).convert("RGB")
        st.image(source, caption="Original image", use_container_width=True)
        if st.button("Transform image", type="primary", disabled=not prompt.strip()):
            try:
                with st.spinner("Transforming image..."):
                    result = edit_image(source, prompt)
                st.image(result, caption="Transformed image", use_container_width=True)
                st.download_button("Download result", result.getvalue(), file_name="edited_image.png", mime="image/png")
            except Exception as e:
                show_error(e)

elif mode == "Chat with PDF (RAG)":
    st.subheader("Chat with your PDF documents")
    st.write("Upload PDFs, index their text locally, and ask questions grounded in the retrieved passages.")
    uploads = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index documents", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting text and building the local search index..."):
                chunks, index, metadata = index_pdfs(uploads)
            st.session_state.rag_chunks = chunks
            st.session_state.rag_index = index
            st.session_state.rag_metadata = metadata
            st.success(f"Indexed {len(chunks)} text chunks from {len(uploads)} PDF(s).")
        except Exception as e:
            show_error(e)

    if "rag_index" in st.session_state:
        question = st.text_input("Ask a question about your documents", key="rag_question")
        if st.button("Get answer", disabled=not question.strip()):
            try:
                with st.spinner("Searching documents and preparing an answer..."):
                    answer, sources = answer_from_pdfs(
                        question,
                        st.session_state.rag_chunks,
                        st.session_state.rag_index,
                        st.session_state.rag_metadata,
                    )
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Retrieved sources")
                if sources:
                    for source in sources:
                        with st.expander(f"{source['filename']} — page {source['page']}"):
                            st.write(source["text"])
                else:
                    st.caption("No source passages were retrieved.")
            except Exception as e:
                show_error(e)

st.divider()
st.caption("Prototype note: hosted providers may charge for usage and may change model availability. Do not upload confidential documents unless you are comfortable with the configured provider's data handling.")

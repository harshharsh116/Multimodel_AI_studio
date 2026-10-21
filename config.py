import os
import streamlit as st

def get_secret(name: str, default=None):
    """Read a secret from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)

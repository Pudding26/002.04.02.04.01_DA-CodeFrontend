import json
import base64
import pathlib
import re
from urllib.parse import quote


def encode_state(state: dict) -> str:
    """Encode dict → base64 string safe for URLs."""
    return base64.urlsafe_b64encode(json.dumps(state).encode()).decode()


def decode_state(encoded: str) -> dict:
    """Decode base64 string back to dict."""
    try:
        return json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    except Exception:
        return {}


def normalize_page_name(path: str) -> str:
    """
    Normalize a page filename into the URL segment Streamlit expects.

    Rules:
      - Strip numeric prefix + underscore (e.g. '02_Visualization' -> 'Visualization')
      - Replace underscores with spaces
      - URL-encode spaces and special chars
    """
    page_stem = pathlib.Path(path).stem   # e.g. "02_Visualization"

    # Strip leading digits + underscore (Streamlit ordering convention)
    page_stem = re.sub(r"^\d+_", "", page_stem)   # -> "Visualization"

    # Streamlit replaces underscores with spaces
    page_name = page_stem.replace("_", " ")

    return quote(page_name)  # URL safe


def build_url(path: str, state: dict, base: str = "http://localhost:59502"):
    """
    Build a shareable Streamlit URL with encoded state.

    Args:
        path (str): Path to this page file (or its name).
        state (dict): Full state dict to encode.
        base (str): Base URL of Streamlit app.

    Returns:
        str: Fully qualified shareable URL.
    """
    query = f"state={quote(encode_state(state))}"
    page_segment = normalize_page_name(path)
    return f"{base}/{page_segment}?{query}"

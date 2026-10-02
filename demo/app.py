"""Streamlit Application for Glass Defect Inspector Demo.

Provides pixel-perfect replica of stitch_image_ui_replica/code.html
connected to real YOLO26m defect inspection checkpoints.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
from demo.server import start_server, is_port_in_use

HTML_PATH = Path(__file__).parent / "ui_component" / "index.html"
BACKEND_PORT = 8502

# Ensure background API server is running on port 8502
@st.cache_resource
def ensure_backend_server():
    if not is_port_in_use(BACKEND_PORT):
        try:
            srv, port = start_server(port=BACKEND_PORT)
            return True
        except Exception as e:
            print(f"Could not start background server on {BACKEND_PORT}: {e}")
            return False
    return True


def main():
    st.set_page_config(
        page_title="Glass Defect Inspector — Model Demo",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Global CSS to remove Streamlit padding and border
    st.markdown(
        """
        <style>
            #MainMenu, header, footer, .stDeployButton {
                visibility: hidden !important;
                display: none !important;
                height: 0 !important;
            }
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            div[data-testid="stVerticalBlock"] {
                gap: 0 !important;
            }
            div[data-testid="stAppViewBlockContainer"] {
                padding: 0 !important;
            }
            html, body, [data-testid="stAppViewContainer"], .main {
                overflow: hidden !important;
                height: 100vh !important;
            }
            iframe {
                width: 100% !important;
                height: 100vh !important;
                border: none !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Start background API server
    ensure_backend_server()

    # Load HTML content directly and embed via st.components.v1.html (no scrolling)
    if HTML_PATH.exists():
        html_content = HTML_PATH.read_text(encoding="utf-8")
        components.html(html_content, height=900, scrolling=False)
    else:
        st.error(f"HTML template not found at {HTML_PATH}")


if __name__ == "__main__":
    main()

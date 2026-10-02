"""Root entrypoint for Glass Defect Inspector Demo.

Can be run either via:
    1) python app.py             (Direct standalone runner, opens browser automatically)
    2) streamlit run app.py      (Streamlit runner)
"""

from __future__ import annotations

import os
import sys
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Detect if running under Streamlit
is_streamlit = "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv)

if is_streamlit:
    from demo.app import main
    if __name__ == "__main__":
        main()
else:
    from demo.server import DefectInspectorHandler, find_available_port

    def run_standalone():
        port = find_available_port(8501)
        url = f"http://localhost:{port}"
        print("=" * 65)
        print(" GLASS DEFECT INSPECTOR — MODEL DEMO")
        print("=" * 65)
        print(f" Server running at: {url}")
        print(" Opening web browser...")
        print(" Press Ctrl+C to stop.")
        print("=" * 65)

        # Launch browser on Windows using OS start command
        try:
            os.system(f"start {url}")
        except Exception:
            webbrowser.open(url)

        server = ThreadingHTTPServer(("127.0.0.1", port), DefectInspectorHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()

    if __name__ == "__main__":
        run_standalone()

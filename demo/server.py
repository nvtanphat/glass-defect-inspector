"""HTTP server for Glass Defect Inspector Demo.

Serves the pixel-perfect HTML UI and provides real inference API endpoints.
Runs standalone or as background server for Streamlit.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference_engine import (
    get_available_checkpoints,
    pil_to_base64,
    run_inspection,
)
from PIL import Image

COMPONENT_DIR = Path(__file__).parent / "ui_component"
HTML_FILE = COMPONENT_DIR / "index.html"
SAMPLES_DIR = Path(__file__).parent / "sample_images"


def get_sample_images_dict() -> dict[str, str]:
    """Load sample images as base64."""
    sample_dict = {}
    mapping = {
        "replica": "sample_replica.jpg",
        "bottle1": "sample_bottle_1.jpg",
        "bottle2": "sample_bottle_2.jpg",
        "bottle3": "sample_bottle_3.jpg",
    }
    for key, filename in mapping.items():
        file_path = SAMPLES_DIR / filename
        if file_path.exists():
            try:
                img = Image.open(file_path).convert("RGB")
                sample_dict[key] = pil_to_base64(img, format="JPEG", quality=88)
            except Exception as e:
                print(f"Error loading sample {filename}: {e}")
    return sample_dict


class DefectInspectorHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("", "/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_error(404, "UI template not found")
                return
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/models":
            models = get_available_checkpoints()
            data = json.dumps({"models": models}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/samples":
            samples = get_sample_images_dict()
            data = json.dumps({"samples": samples}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/health":
            data = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                req = json.loads(body)

                image_input = req.get("image_data") or req.get("image")
                if not image_input and req.get("sample_key"):
                    sample_dict = get_sample_images_dict()
                    image_input = sample_dict.get(req.get("sample_key"))

                weights_path = req.get("weights_path") or req.get("model")
                conf = float(req.get("conf_threshold", req.get("conf", 0.25)))
                iou = float(req.get("iou_threshold", req.get("iou", 0.45)))
                nms_mode = req.get("nms_mode", "nms_free")
                enable_roi = bool(req.get("enable_roi", True))
                enable_tiling = bool(req.get("enable_tiling", False))
                tile_size = int(req.get("tile_size", 960))
                overlap_ratio = float(req.get("overlap_ratio", 0.20))

                # Resolve weights path if relative or name
                if weights_path and not Path(weights_path).is_absolute():
                    resolved = PROJECT_ROOT / "weights" / weights_path
                    if resolved.exists():
                        weights_path = str(resolved)
                    else:
                        # Try matching with available checkpoints
                        for ckpt in get_available_checkpoints():
                            if ckpt["filename"] == weights_path or ckpt["name"] == weights_path:
                                weights_path = ckpt["id"]
                                break

                if not weights_path or not Path(weights_path).exists():
                    weights_path = str(PROJECT_ROOT / "weights" / "yolo26m_roi_vfl_best.pt")

                result = run_inspection(
                    image_input=image_input,
                    weights_path=weights_path,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    nms_mode=nms_mode,
                    enable_roi=enable_roi,
                    enable_tiling=enable_tiling,
                    tile_size=tile_size,
                    overlap_ratio=overlap_ratio,
                )

                resp_bytes = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(resp_bytes)

            except Exception as e:
                import traceback
                traceback.print_exc()
                err_data = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_data)))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(err_data)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Quiet logging
        pass


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8501, max_tries: int = 20) -> int:
    for p in range(start_port, start_port + max_tries):
        if not is_port_in_use(p):
            return p
    return start_port


def start_server(port: int = 8501, host: str = "127.0.0.1") -> tuple[ThreadingHTTPServer, int]:
    actual_port = find_available_port(port)
    server = ThreadingHTTPServer((host, actual_port), DefectInspectorHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, actual_port


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = find_available_port(args.port)
    print(f"Starting Glass Defect Inspector Demo Server on http://localhost:{port}...")
    server, port = start_server(port)
    url = f"http://localhost:{port}"
    print(f"Server ready at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()

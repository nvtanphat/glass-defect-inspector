import os
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel = sys.argv[1] if len(sys.argv) > 1 else "phtnguyn1ytj/scale-error-yolo26-sahi-train"
    out_dir = Path("kaggle_logs")
    out_dir.mkdir(exist_ok=True)
    res = api.kernels_output(kernel, path=str(out_dir))
    print("OUTPUT RESULT:", res)
    for p in out_dir.iterdir():
        print(f"File: {p.name}, size: {p.stat().st_size} bytes")
        if p.name.endswith(".log") or p.stat().st_size < 100000:
            print("--- Content preview ---")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                print(text[-2000:])
            except Exception as e:
                print(f"Read error: {e}")

if __name__ == "__main__":
    main()

import json
import os
import requests
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def inspect_kernel(slug):
    print(f"=== Inspecting {slug} ===")
    url = "https://www.kaggle.com/api/v1/kernels/output"
    r = requests.get(url, auth=auth, params={"userName": "phtnguyn1ytj", "kernelSlug": slug}, timeout=60)
    data = r.json()
    files = data.get("files", [])
    print(f"Found {len(files)} output files:")
    for f in files:
        print(f"  {f.get('fileName')}: {f.get('totalBytes', 0)} bytes, url: {f.get('url')}")
    
    log_raw = data.get("logNullable") or data.get("log") or ""
    if log_raw:
        try:
            entries = json.loads(log_raw)
            # Find lines mentioning mAP or metrics or evaluation
            print("\n--- Evaluation / Metric snippets from log ---")
            for e in entries:
                text = e.get("data", "")
                if any(k in text.lower() for k in ["map50", "map50-95", "metrics", "divat", "loichi", "loinhua", "loitray", "all"]):
                    print(text.strip())
        except Exception as ex:
            print("Error parsing log json:", ex)

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "scale-error-yolo26-roi-crop-train"
    inspect_kernel(slug)

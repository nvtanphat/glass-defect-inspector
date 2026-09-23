import json
import os
import requests
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KAGGLE_USERNAME"] = "jyang10"
os.environ["KAGGLE_KEY"] = "083346fbb9da8ad61b964e678d18a71e"

def print_log(kernel_slug):
    url = "https://www.kaggle.com/api/v1/kernels/output"
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    params = {"userName": "jyang10", "kernelSlug": kernel_slug}
    r = requests.get(url, auth=auth, params=params, timeout=30)
    data = r.json()
    log_raw = data.get("logNullable") or data.get("log")
    if log_raw:
        try:
            entries = json.loads(log_raw)
            for item in entries:
                stream = item.get("stream_name", "")
                text = item.get("data", "")
                print(f"[{stream}] {text}", end="")
        except Exception:
            print(log_raw)
    else:
        print("No log available")

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "scale-error-yolo26-sahi-train"
    print_log(slug)

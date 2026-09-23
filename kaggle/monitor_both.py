import json
import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KAGGLE_USERNAME"] = "jyang10"
os.environ["KAGGLE_KEY"] = "083346fbb9da8ad61b964e678d18a71e"

KERNELS = [
    "scale-error-sahi-finetune",
    "scale-error-sahi-study"
]

def get_status(slug):
    url = f"https://www.kaggle.com/api/v1/kernels/status"
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    params = {"userName": "jyang10", "kernelSlug": slug}
    r = requests.get(url, auth=auth, params=params, timeout=20)
    if r.status_code == 200:
        return r.json().get("status", "UNKNOWN")
    return f"HTTP_{r.status_code}"

def get_log(slug):
    url = f"https://www.kaggle.com/api/v1/kernels/output"
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    params = {"userName": "jyang10", "kernelSlug": slug}
    r = requests.get(url, auth=auth, params=params, timeout=20)
    if r.status_code == 200:
        data = r.json()
        log_raw = data.get("logNullable") or data.get("log")
        if log_raw:
            try:
                entries = json.loads(log_raw)
                lines = []
                for item in entries:
                    stream = item.get("stream_name", "")
                    text = item.get("data", "")
                    lines.append(f"[{stream}] {text.strip()}")
                return "\n".join(lines[-15:])
            except Exception:
                return str(log_raw)[-500:]
    return "(No log)"

def monitor(iterations=8, interval=20):
    for i in range(1, iterations + 1):
        print(f"\n================ [Check #{i}/{iterations} - {time.strftime('%H:%M:%S')}] ================", flush=True)
        for slug in KERNELS:
            st = get_status(slug)
            print(f">>> Kernel: {slug} | Status: {st}", flush=True)
            log_tail = get_log(slug)
            if log_tail and log_tail != "(No log)":
                print(f"--- Log tail ({slug}) ---\n{log_tail}\n", flush=True)
        if i < iterations:
            time.sleep(interval)

if __name__ == "__main__":
    monitor(iterations=6, interval=25)

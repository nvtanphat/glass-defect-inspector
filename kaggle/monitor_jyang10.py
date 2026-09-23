import json
import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

auth = ("jyang10", "083346fbb9da8ad61b964e678d18a71e")

def check_kernel(slug: str):
    st_res = requests.get(
        "https://www.kaggle.com/api/v1/kernels/status",
        auth=auth,
        params={"userName": "jyang10", "kernelSlug": slug},
        timeout=15,
    )
    data = st_res.json()
    st = data.get("status", "UNKNOWN")
    msg = data.get("failureMessage", "")

    out_res = requests.get(
        "https://www.kaggle.com/api/v1/kernels/output",
        auth=auth,
        params={"userName": "jyang10", "kernelSlug": slug},
        timeout=15,
    )
    raw = out_res.json().get("logNullable") or ""
    print(f"[{time.strftime('%H:%M:%S')}] {slug}: {st} (failure: {msg or 'None'}), log_len: {len(raw)}", flush=True)

    if len(raw) > 50:
        try:
            entries = json.loads(raw)
            print(f"--- TAIL LOG ({slug}) ---", flush=True)
            for e in entries[-12:]:
                stream = e.get("stream_name", "")
                d = e.get("data", "").strip()
                print(f"[{stream}] {d}", flush=True)
        except Exception:
            print(raw[-500:], flush=True)

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "scale-error-sahi-finetune"
    check_kernel(slug)

import json
import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")
slugs = ["scale-error-yolo26-roi-crop-train", "scale-error-yolo26-sahi-train"]

def check_once():
    for slug in slugs:
        st_res = requests.get(
            "https://www.kaggle.com/api/v1/kernels/status",
            auth=auth,
            params={"userName": "phtnguyn1ytj", "kernelSlug": slug},
            timeout=15,
        )
        st = st_res.json().get("status", "UNKNOWN")
        msg = st_res.json().get("failureMessage", "")

        out_res = requests.get(
            "https://www.kaggle.com/api/v1/kernels/output",
            auth=auth,
            params={"userName": "phtnguyn1ytj", "kernelSlug": slug},
            timeout=15,
        )
        raw = out_res.json().get("logNullable") or ""
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: {st} (fail: {msg or 'None'}), log_len: {len(raw)}", flush=True)

        if len(raw) > 50:
            try:
                entries = json.loads(raw)
                print(f"--- TAIL LOG ({slug}) ---", flush=True)
                for e in entries[-8:]:
                    stream = e.get("stream_name", "")
                    data = e.get("data", "").strip()
                    print(f"[{stream}] {data}", flush=True)
            except Exception:
                print(raw[-500:], flush=True)

if __name__ == "__main__":
    check_once()

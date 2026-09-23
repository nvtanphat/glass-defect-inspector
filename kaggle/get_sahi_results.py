import json
import os
import requests
from pathlib import Path

auth = ("jyang10", "083346fbb9da8ad61b964e678d18a71e")

def inspect_sahi_eval():
    url = "https://www.kaggle.com/api/v1/kernels/output"
    r = requests.get(url, auth=auth, params={"userName": "jyang10", "kernelSlug": "scale-error-sahi-eval"}, timeout=60)
    data = r.json()
    files = data.get("files", [])
    print(f"Total output files: {len(files)}")
    out_dir = Path("kaggle_sahi_eval_results")
    out_dir.mkdir(exist_ok=True)
    
    for f in files:
        name = f.get("fileName")
        print(f"File: {name} ({f.get('totalBytes')} bytes)")
        if name and ("json" in name or "csv" in name or "metrics" in name):
            file_url = f.get("url")
            fr = requests.get(file_url, auth=auth, timeout=30)
            target = out_dir / Path(name).name
            target.write_bytes(fr.content)
            print(f"Downloaded {target.name}:")
            try:
                print(json.dumps(json.loads(target.read_text(encoding="utf-8")), indent=2))
            except Exception as e:
                print(target.read_text(encoding="utf-8")[:500])
                
    log_raw = data.get("logNullable") or data.get("log") or ""
    if log_raw:
        try:
            entries = json.loads(log_raw)
            print("\n--- Log Highlights ---")
            for e in entries:
                text = e.get("data", "")
                if any(k in text.lower() for k in ["sahi evaluation", "metrics summary", "map", "complete", "divat", "loitray"]):
                    print(text.strip())
        except Exception:
            print("Log preview:", log_raw[-1500:])

if __name__ == "__main__":
    inspect_sahi_eval()

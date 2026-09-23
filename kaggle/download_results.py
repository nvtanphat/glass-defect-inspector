import os
import requests
import json
from pathlib import Path

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def download_metrics():
    url = "https://www.kaggle.com/api/v1/kernels/output"
    r = requests.get(url, auth=auth, params={"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-roi-crop-train"}, timeout=60)
    data = r.json()
    files = data.get("files", [])
    
    out_dir = Path("kaggle_eval_results")
    out_dir.mkdir(exist_ok=True)
    
    for f in files:
        name = f.get("fileName")
        if name and ("metrics" in name or "results.csv" in name or "args.yaml" in name):
            file_url = f.get("url")
            print(f"Downloading {name}...")
            fr = requests.get(file_url, auth=auth, timeout=60)
            local_path = out_dir / Path(name).name
            local_path.write_bytes(fr.content)
            print(f"Saved {local_path} ({len(fr.content)} bytes)")
            if local_path.suffix == ".json":
                try:
                    content = json.loads(local_path.read_text(encoding="utf-8"))
                    print(f"--- Content of {local_path.name} ---")
                    print(json.dumps(content, indent=2))
                except Exception as e:
                    print("Error parsing json:", e)

if __name__ == "__main__":
    download_metrics()

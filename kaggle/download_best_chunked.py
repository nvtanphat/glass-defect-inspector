import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def download_best():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    url = "https://www.kaggle.com/api/v1/kernels/output"
    r = session.get(url, auth=auth, params={"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-roi-crop-train"})
    files = r.json().get("files", [])
    best_f = None
    for f in files:
        if f.get("fileName", "").endswith("best.pt"):
            best_f = f
            break
            
    if not best_f:
        print("best.pt not found")
        return

    print("Target:", best_f.get("fileName"), "url:", best_f.get("url")[:60])
    resp = session.get(best_f.get("url"), auth=auth, stream=True, timeout=60)
    print("Download HTTP status:", resp.status_code)
    
    out_path = "kaggle_eval_results/best.pt"
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded: {downloaded / 1024 / 1024:.2f} MB", end="\r")
    print(f"\nDone! Saved {out_path} ({downloaded} bytes)")

if __name__ == "__main__":
    download_best()

import os
import requests
import time

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def download_best_resumable():
    url = "https://www.kaggle.com/api/v1/kernels/output"
    r = requests.get(url, auth=auth, params={"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-roi-crop-train"})
    files = r.json().get("files", [])
    best_f = None
    for f in files:
        if f.get("fileName", "").endswith("best.pt"):
            best_f = f
            break
            
    if not best_f:
        print("best.pt not found")
        return

    out_path = "kaggle_eval_results/best.pt"
    
    # Check current downloaded size
    existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"Existing bytes: {existing / 1024 / 1024:.2f} MB")
    
    # Download loop with Range header
    max_attempts = 15
    for attempt in range(max_attempts):
        existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            
        print(f"\nAttempt {attempt+1}/{max_attempts}, starting at {existing / 1024 / 1024:.2f} MB...")
        try:
            resp = requests.get(best_f.get("url"), auth=auth, headers=headers, stream=True, timeout=30)
            if resp.status_code not in [200, 206]:
                print(f"Unexpected status: {resp.status_code}")
                # Refresh URL if expired
                r = requests.get(url, auth=auth, params={"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-roi-crop-train"})
                best_f = [f for f in r.json().get("files", []) if f.get("fileName", "").endswith("best.pt")][0]
                continue
                
            mode = "ab" if existing > 0 and resp.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0
                
            with open(out_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
                        existing += len(chunk)
                        print(f"Progress: {existing / 1024 / 1024:.2f} MB", end="\r", flush=True)
                        
            print(f"\nStream ended. File size now: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
            # If size > 40MB, it's complete!
            if os.path.getsize(out_path) > 42 * 1024 * 1024:
                print("Download complete and verified!")
                break
        except Exception as e:
            print(f"\nConnection hiccup: {e}. Retrying in 2s...")
            time.sleep(2)

if __name__ == "__main__":
    download_best_resumable()

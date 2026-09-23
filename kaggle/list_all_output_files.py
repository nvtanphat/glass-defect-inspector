import os
import requests

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def list_all_files():
    url = "https://www.kaggle.com/api/v1/kernels/output"
    token = None
    all_files = []
    while True:
        params = {"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-sahi-train"}
        if token:
            params["pageToken"] = token
        r = requests.get(url, auth=auth, params=params, timeout=20)
        data = r.json()
        files = data.get("files", [])
        all_files.extend(files)
        token = data.get("nextPageToken")
        if not token or len(files) == 0:
            break
            
    print(f"Total files across all pages: {len(all_files)}")
    for f in all_files:
        name = f.get("fileName", "")
        if any(k in name for k in [".pt", "runs", "weights", "results", "args", "sahi", "metrics", "csv"]):
            print("FOUND:", name, f.get("totalBytes"))

if __name__ == "__main__":
    list_all_files()

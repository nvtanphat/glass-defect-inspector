import os
import requests

auth = ("phtnguyn1ytj", "16975e4a229dab67962c3f165b86feb1")

def test_versions():
    # 1. Try querying kernels/output with versionNumber
    for v in [3, 4, None]:
        url = "https://www.kaggle.com/api/v1/kernels/output"
        params = {"userName": "phtnguyn1ytj", "kernelSlug": "scale-error-yolo26-sahi-train"}
        if v:
            params["versionNumber"] = v
        r = requests.get(url, auth=auth, params=params, timeout=15)
        print(f"v={v} status={r.status_code}, keys={list(r.json().keys()) if r.status_code==200 else r.text[:200]}")
        if r.status_code == 200:
            data = r.json()
            raw_log = data.get("logNullable") or data.get("log") or ""
            files = data.get("files", [])
            print(f"   log_len={len(raw_log)}, files={len(files)}")
            if len(files) > 0:
                for f in files:
                    print(f"      file: {f.get('fileName')}")

if __name__ == "__main__":
    test_versions()

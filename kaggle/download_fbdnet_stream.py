import base64
import json
import os
from pathlib import Path
import time
import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

ACCOUNTS = json.loads((Path(__file__).parent / ".kaggle_accounts.json").read_text(encoding="utf-8"))
os.environ["KAGGLE_USERNAME"] = ACCOUNTS["baohuynh05"]["username"]
os.environ["KAGGLE_KEY"] = ACCOUNTS["baohuynh05"]["key"]

api = KaggleApi()
api.authenticate()

resume_dir = Path("kaggle/dataset_fbdnet_resume69")
resume_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path("kaggle-output/scale-error-dfine-fbdnet-m-resume46")
out_dir.mkdir(parents=True, exist_ok=True)

print("Fetching URLs for output_fbdnet_m files...", flush=True)

with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "baohuynh05"
    req.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
    api._set_paging(req, 1)
    res = client.kernels.kernels_api_client.list_kernel_session_output(req)
    prefix = json.loads(base64.b64decode(res.next_page_token).decode())["GcsPageToken"].split("/output/")[0]
    
    target_token = base64.b64encode(json.dumps({"GcsPageToken": f"{prefix}/output/output_fbdnet_m"}).encode()).decode()
    req2 = ApiListKernelSessionOutputRequest()
    req2.user_name = "baohuynh05"
    req2.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
    api._set_paging(req2, 200, target_token)
    res2 = client.kernels.kernels_api_client.list_kernel_session_output(req2)
    file_map = {f.file_name: f.url for f in (res2.files or [])}

print(f"Available files: {list(file_map.keys())}", flush=True)


def download_stream(url: str, dest_path: Path):
    print(f"\nDownloading {dest_path.name}...", flush=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".download")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=(20, 60)) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB", flush=True)
        downloaded = 0
        last_log = t0
        with open(temp_path, "wb") as fp:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    fp.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_log >= 5:
                        speed = (downloaded / (1024 * 1024)) / (time.time() - t0)
                        pct = (downloaded / total_size * 100) if total_size else 0
                        print(f"  {downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB ({pct:.1f}%) - {speed:.1f} MB/s", flush=True)
                        last_log = time.time()

    temp_path.replace(dest_path)
    print(f"Finished {dest_path.name} in {time.time() - t0:.1f}s!", flush=True)


# 1. Download last.pth
if "output_fbdnet_m/last.pth" in file_map:
    target = resume_dir / "last.pth"
    if not target.exists() or target.stat().st_size < 300 * 1024 * 1024:
        download_stream(file_map["output_fbdnet_m/last.pth"], target)
    else:
        print("last.pth already downloaded.")

# 2. Download best_stg1.pth
if "output_fbdnet_m/best_stg1.pth" in file_map:
    target = resume_dir / "best_stg1.pth"
    if not target.exists() or target.stat().st_size < 300 * 1024 * 1024:
        download_stream(file_map["output_fbdnet_m/best_stg1.pth"], target)
    else:
        print("best_stg1.pth already downloaded.")

print("\nAll downloads complete!")

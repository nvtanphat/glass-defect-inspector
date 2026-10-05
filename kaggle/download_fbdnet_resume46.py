import json
import os
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi

ACCOUNTS = json.loads((Path(__file__).parent / ".kaggle_accounts.json").read_text(encoding="utf-8"))

os.environ["KAGGLE_USERNAME"] = ACCOUNTS["baohuynh05"]["username"]
os.environ["KAGGLE_KEY"] = ACCOUNTS["baohuynh05"]["key"]
api = KaggleApi()
api.authenticate()

target_dir = Path("kaggle-output/scale-error-dfine-fbdnet-m-resume46")
target_dir.mkdir(parents=True, exist_ok=True)

print("Listing session output files for baohuynh05/scale-error-dfine-fbdnet-m-resume46...", flush=True)

with api.build_kaggle_client() as client:
    token = None
    matched_files = []
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "baohuynh05"
        req.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
        api._set_paging(req, 200, token)
        res = client.kernels.kernels_api_client.list_kernel_session_output(req)
        for f in (res.files or []):
            fn = f.file_name
            if "output_fbdnet_m" in fn or fn.endswith(".pth") or fn.endswith(".txt") or fn.endswith(".json"):
                if "data_roi" not in fn and ".git" not in fn and "coco" not in fn:
                    matched_files.append((fn, f.url))
        token = getattr(res, "next_page_token", None)
        if not token:
            break

print(f"Found {len(matched_files)} relevant files:", flush=True)
for fn, url in matched_files:
    print(f"  {fn}", flush=True)

for fn, url in matched_files:
    dest = target_dir / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {fn}", flush=True)
        continue
    print(f"Downloading {fn}...", flush=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as fp:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    fp.write(chunk)
    print(f"Saved: {fn} ({dest.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)

print("Download complete!", flush=True)

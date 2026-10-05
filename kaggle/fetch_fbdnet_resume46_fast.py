import base64
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

output_dir = Path("kaggle-output/scale-error-dfine-fbdnet-m-resume46")
output_dir.mkdir(parents=True, exist_ok=True)

resume_dataset_dir = Path("kaggle/dataset_fbdnet_resume69")
resume_dataset_dir.mkdir(parents=True, exist_ok=True)

with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "baohuynh05"
    req.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
    api._set_paging(req, 1)
    res = client.kernels.kernels_api_client.list_kernel_session_output(req)
    t = res.next_page_token
    d = json.loads(base64.b64decode(t).decode())
    prefix = d["GcsPageToken"].split("/output/")[0]
    
    target_token = base64.b64encode(json.dumps({"GcsPageToken": f"{prefix}/output/output_fbdnet_m"}).encode()).decode()
    req2 = ApiListKernelSessionOutputRequest()
    req2.user_name = "baohuynh05"
    req2.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
    api._set_paging(req2, 200, target_token)
    res2 = client.kernels.kernels_api_client.list_kernel_session_output(req2)
    
    files_to_download = [
        "output_fbdnet_m/best_stg1.pth",
        "output_fbdnet_m/last.pth",
        "output_fbdnet_m/log.txt",
        "output_fbdnet_m/train_stdout.log",
    ]
    
    for f in (res2.files or []):
        if f.file_name in files_to_download:
            dest1 = output_dir / f.file_name
            dest1.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {f.file_name}...", flush=True)
            with requests.get(f.url, stream=True) as r:
                r.raise_for_status()
                with open(dest1, "wb") as fp:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            fp.write(chunk)
            print(f"Saved to {dest1} ({dest1.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
            
            # Also copy to dataset_fbdnet_resume69
            basename = Path(f.file_name).name
            if basename in ["best_stg1.pth", "last.pth", "log.txt"]:
                dest2 = resume_dataset_dir / basename
                import shutil
                shutil.copy2(dest1, dest2)
                print(f"Copied {basename} to {dest2}", flush=True)

print("All fbdnet resume files fetched successfully!", flush=True)

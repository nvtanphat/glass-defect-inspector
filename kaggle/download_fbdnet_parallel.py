import base64
import concurrent.futures
import json
import os
from pathlib import Path
import re
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

# 1. Download log.txt immediately
if "output_fbdnet_m/log.txt" in file_map:
    r = requests.get(file_map["output_fbdnet_m/log.txt"])
    (resume_dir / "log.txt").write_bytes(r.content)
    (out_dir / "output_fbdnet_m").mkdir(parents=True, exist_ok=True)
    (out_dir / "output_fbdnet_m/log.txt").write_bytes(r.content)
    print(f"Saved log.txt ({len(r.content):,} bytes)", flush=True)


def download_file_parallel(url: str, dest_path: Path, workers: int = 8, chunk_size: int = 16 * 1024 * 1024):
    print(f"\nStarting parallel download for {dest_path.name}...", flush=True)
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=(30, 90))
    probe.raise_for_status()
    match = re.search(r"/(\d+)$", probe.headers.get("Content-Range", ""))
    total_size = int(match.group(1))
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB", flush=True)

    parts_dir = dest_path.with_suffix(dest_path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    
    ranges = [
        (idx, start, min(start + chunk_size - 1, total_size - 1))
        for idx, start in enumerate(range(0, total_size, chunk_size))
    ]

    def fetch_chunk(spec):
        idx, start, end = spec
        part_file = parts_dir / f"{idx:05d}.part"
        expected = end - start + 1
        if part_file.exists() and part_file.stat().st_size == expected:
            return part_file
        for attempt in range(1, 6):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 90)) as resp:
                    resp.raise_for_status()
                    with open(part_file, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fp.write(chunk)
                if part_file.stat().st_size == expected:
                    return part_file
            except Exception as e:
                if attempt == 5:
                    raise
        return part_file

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_chunk, spec) for spec in ranges]
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done_count += 1
            if done_count % 5 == 0 or done_count == len(ranges):
                print(f"  Downloaded {done_count}/{len(ranges)} chunks ({done_count * chunk_size / (1024 * 1024):.1f} MB)...", flush=True)

    # Stitch parts
    print(f"Assembling {dest_path.name}...", flush=True)
    with open(dest_path, "wb") as out_fp:
        for idx in range(len(ranges)):
            part_file = parts_dir / f"{idx:05d}.part"
            with open(part_file, "rb") as p_fp:
                out_fp.write(p_fp.read())
            part_file.unlink()
    parts_dir.rmdir()
    print(f"Successfully assembled {dest_path.name} ({dest_path.stat().st_size / (1024 * 1024):.1f} MB)!", flush=True)


# Download last.pth (priority for resuming training)
if "output_fbdnet_m/last.pth" in file_map:
    dest_last = resume_dir / "last.pth"
    download_file_parallel(file_map["output_fbdnet_m/last.pth"], dest_last)
    # copy to output_dir
    import shutil
    shutil.copy2(dest_last, out_dir / "output_fbdnet_m/last.pth")

# Download best_stg1.pth
if "output_fbdnet_m/best_stg1.pth" in file_map:
    dest_best = resume_dir / "best_stg1.pth"
    download_file_parallel(file_map["output_fbdnet_m/best_stg1.pth"], dest_best)
    import shutil
    shutil.copy2(dest_best, out_dir / "output_fbdnet_m/best_stg1.pth")

print("\nAll checkpoints and logs successfully downloaded and ready for resume69!", flush=True)

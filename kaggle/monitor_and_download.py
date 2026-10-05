"""Monitor and automatically download outputs for Kaggle kernels across accounts.

Monitored kernels:
1. leppan/scale-error-dfine-m-tiled (RUNNING on leppan)
2. jyang10/scale-error-dfine-m-resume-71-100 (RUNNING on jyang10)
3. baohuynh05/scale-error-dfine-m-resume-47-70 (COMPLETE, downloaded)
4. baohuynh05/scale-error-dfine-m-resume-24-46 (COMPLETE, downloaded)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "kaggle-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS_FILE = Path(__file__).parent / ".kaggle_accounts.json"
if ACCOUNTS_FILE.exists():
    ACCOUNTS = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
else:
    ACCOUNTS = {
        "baohuynh05": {"username": "baohuynh05", "key": os.getenv("KAGGLE_KEY_BAOHUYNH05", "")},
        "jyang10": {"username": "jyang10", "key": os.getenv("KAGGLE_KEY_JYANG10", "")},
        "leppan": {"username": os.getenv("KAGGLE_USERNAME", "leppan"), "key": os.getenv("KAGGLE_KEY", "")},
    }


TARGET_KERNELS = [
    "jyang10/scale-error-dfine-m-baseline-resume69",
    "phtnguyn1ytj/scale-error-dfine-official-clone",
    "baohuynh05/scale-error-dfine-fbdnet-m-resume46",
    "jyang10/scale-error-dfine-m-baseline-resume47",
    "baohuynh05/scale-error-dfine-m-baseline-resume23",
    "leppan/scale-error-dfine-fbdnet-m-resume23",
    "baohuynh05/scale-error-dfine-m-baseline",
    "leppan/scale-error-dfine-fbdnet-m",
    "jyang10/scale-error-dfine-frequency-official",
    "jyang10/scale-error-dfine-exp-frequency-screening",
    "jyang10/scale-error-dfine-m-resume-71-100",
    "leppan/scale-error-dfine-m-resume-71-100",
    "leppan/scale-error-dfine-exp-p2-screening",
    "baohuynh05/scale-error-dfine-m-resume-47-70",
    "baohuynh05/scale-error-dfine-m-resume-24-46",
]



def get_api_for_owner(owner: str):
    from kaggle.api.kaggle_api_extended import KaggleApi

    creds = ACCOUNTS.get(owner, ACCOUNTS["leppan"])
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

    api = KaggleApi()
    api.authenticate()
    return api


def check_and_download(kernel_slug: str) -> dict[str, Any]:
    owner, slug = kernel_slug.split("/")
    api = get_api_for_owner(owner)

    try:
        status_obj = api.kernels_status(kernel_slug)
        raw_status = getattr(status_obj, "status", status_obj)
        status_str = getattr(raw_status, "name", str(raw_status))
        failure_msg = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
    except Exception as e:
        status_str = "NOT_FOUND"
        failure_msg = str(e)

    result = {
        "kernel": kernel_slug,
        "status": status_str,
        "failure_message": failure_msg,
        "downloaded": False,
        "download_path": "",
    }

    if status_str == "NOT_FOUND":
        return result

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {kernel_slug} -> {status_str}")

    target_path = OUTPUT_DIR / slug

    if status_str in {"COMPLETE", "KernelWorkerStatus.COMPLETE", "CANCEL_ACKNOWLEDGED", "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"}:
        done_marker = target_path / ".download_done"
        if done_marker.exists():
            print(f" -> Kernel {kernel_slug} is {status_str} and already downloaded.")
            result["downloaded"] = True
            result["download_path"] = str(target_path)
            return result

        print(f" -> Kernel {kernel_slug} is {status_str}! Downloading output to {target_path}...")
        target_path.mkdir(parents=True, exist_ok=True)
        import requests
        from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
        with api.build_kaggle_client() as client:
            token = None
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = owner
                req.kernel_slug = slug
                api._set_paging(req, 200, token)
                res = client.kernels.kernels_api_client.list_kernel_session_output(req)
                for f in (res.files or []):
                    fn = f.file_name
                    dest = target_path / fn
                    url = getattr(f, "url", None)
                    # Only download model checkpoints, logs, csv, and json metrics
                    ext = dest.suffix.lower()
                    if ext not in {".pth", ".json", ".csv", ".txt", ".yaml", ".yml"}:
                        continue
                    if not url or ".git" in fn or "__pycache__" in fn or "dataset" in fn or "artifacts" in fn or "site-packages" in fn or "data_roi" in fn or "coco" in fn:
                        continue
                    if dest.exists() and dest.stat().st_size > 0:
                        continue
                    print(f"   Downloading {fn}...")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(dest, "wb") as fp:
                            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                                if chunk:
                                     fp.write(chunk)
                    print(f"   -> Saved: {fn} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
                token = getattr(res, "next_page_token", None)
                if not token:
                    break
        done_marker.write_text("done", encoding="utf-8")
        result["downloaded"] = True
        result["download_path"] = str(target_path)

        # Auto-copy key weights to weights/ directory
        if "resume-71-100" in kernel_slug:
            best_pth = target_path / "resume71_best_stg1.pth"
            if best_pth.exists():
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_resume71_best.pth"
                shutil.copy2(best_pth, weights_dest)
                print(f"   -> Successfully copied final weights to {weights_dest}")

        if "frequency-official" in kernel_slug:
            candidates = list(target_path.glob("*best*.pth"))
            if candidates:
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_frequency_official_best.pth"
                shutil.copy2(candidates[0], weights_dest)
                print(f"   -> Successfully copied frequency official best weights to {weights_dest}")
        elif "frequency" in kernel_slug:
            candidates = list(target_path.glob("*best*.pth"))
            if candidates:
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_frequency_best.pth"
                shutil.copy2(candidates[0], weights_dest)
                print(f"   -> Successfully copied frequency best weights to {weights_dest}")

        if "scale-error-dfine-m-baseline" in kernel_slug:
            candidates = list(target_path.glob("*best_stg2*.pth")) or list(target_path.glob("*best*.pth"))
            if candidates:
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_baseline_reproduced_best.pth"
                shutil.copy2(candidates[0], weights_dest)
                print(f"   -> Successfully copied baseline reproduced weights to {weights_dest}")

        if "fbdnet-m" in kernel_slug:
            candidates = list(target_path.glob("*best*.pth"))
            if candidates:
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_fbdnet_best.pth"
                shutil.copy2(candidates[0], weights_dest)
                print(f"   -> Successfully copied FBDNet-M best weights to {weights_dest}")

        if "scale-error-dfine-m-tiled" in kernel_slug:
            candidates = list(target_path.glob("*best*.pth"))
            if candidates:
                weights_dest = PROJECT_ROOT / "weights" / "dfine_m_tiled_best.pth"
                shutil.copy2(candidates[0], weights_dest)
                print(f"   -> Successfully copied tiled weights to {weights_dest}")

    elif status_str in {
        "FAILED", "KernelWorkerStatus.FAILED",
        "ERROR", "KernelWorkerStatus.ERROR",
        "CANCEL_ACKNOWLEDGED", "KernelWorkerStatus.CANCEL_ACKNOWLEDGED",
        "CANCELLED", "KernelWorkerStatus.CANCELLED",
    }:
        print(f" -> Kernel {kernel_slug} STOPPED/CANCELLED/FAILED! Status: {status_str}, Reason: {failure_msg}")
    else:
        print(f" -> Kernel {kernel_slug} is still running...")

    return result


def monitor_until_complete(target_slug: str = "jyang10/scale-error-dfine-frequency-official", poll_interval: int = 3600):
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        poll_interval = int(sys.argv[1])
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting tracking for {target_slug} (interval: {poll_interval}s)...", flush=True)
    while True:
        res = check_and_download(target_slug)
        status = res.get("status", "")
        if status in {"COMPLETE", "KernelWorkerStatus.COMPLETE"}:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Target kernel completed and downloaded successfully!", flush=True)
            break
        elif status in {"FAILED", "KernelWorkerStatus.FAILED", "ERROR", "KernelWorkerStatus.ERROR", "CANCELLED", "KernelWorkerStatus.CANCELLED"}:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Target kernel ended with status {status}!", flush=True)
            break
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sleeping {poll_interval}s until next check...", flush=True)
        time.sleep(poll_interval)


def check_all_active():
    active_targets = [
        "baohuynh05/scale-error-dfine-fbdnet-m-resume46",
        "phtnguyn1ytj/scale-error-dfine-official-clone",
        "jyang10/scale-error-dfine-m-baseline-resume69",
    ]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking all {len(active_targets)} active kernels...", flush=True)
    results = {}
    for target in active_targets:
        res = check_and_download(target)
        results[target] = res
    return results


if __name__ == "__main__":
    if "--once" in sys.argv or "--check-all" in sys.argv:
        check_all_active()
    else:
        monitor_until_complete()




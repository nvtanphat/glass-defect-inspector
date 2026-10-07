"""Automated Kaggle session monitor and auto-resumer for scale-error models.

Monitors active Kaggle training kernels. When a session ends ("hết phiên" / timeout / cancel / complete)
before finishing all 100 epochs, this script automatically extracts the latest checkpoint URLs
and launches a new resume kernel under the target account (nguynvntnpht).

Checks periodically every 2 hours (7200 seconds) by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
OUTPUT_DIR = PROJECT_ROOT / "kaggle-output"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS_FILE = KAGGLE_DIR / ".kaggle_accounts.json"
STATE_FILE = KAGGLE_DIR / "auto_resume_state.json"
LOG_FILE = KAGGLE_DIR / "auto_resume_monitor.log"

RESUME_ACCOUNT_NAME = "qucthai"

DEFAULT_MODELS = [
    {
        "id": "fbdnet_m",
        "name": "FBDNet-M",
        "current_slug": "phtnguyn1ytj/scale-error-dfine-fbdnet-m-resume69",
        "type": "fbdnet",
        "template_script": KAGGLE_DIR / "kernel_dfine_fbdnet_resume69_pht" / "run_dfine_fbdnet_m.py",
        "code_filename": "run_dfine_fbdnet_m.py",
        "output_subdir": "output_fbdnet_m",
        "metrics_filename": "metrics_fbdnet_m.json",
        "best_dest": WEIGHTS_DIR / "dfine_m_fbdnet_best.pth",
        "slug_prefix": "scale-error-dfine-fbdnet-m-resume",
        "dataset_source": "nguynvntnpht/scale-error-dc-v26-yolov8",
        "status": "RUNNING",
    },
    {
        "id": "dfine_official_clone",
        "name": "D-FINE-M Official Clone",
        "current_slug": "phtnguyn1ytj/scale-error-dfine-official-clone-resume24",
        "type": "official_clone",
        "template_script": KAGGLE_DIR / "kernel_dfine_official_clone_resume24" / "run_dfine.py",
        "code_filename": "run_dfine.py",
        "output_subdir": "output_dfine_m",
        "metrics_filename": "metrics_dfine_m.json",
        "best_dest": WEIGHTS_DIR / "dfine_m_official_clone_best.pth",
        "slug_prefix": "scale-error-dfine-official-clone-resume",
        "dataset_source": "nguynvntnpht/scale-error-dc-v26-yolov8",
        "status": "RUNNING",
    },
]


def log(msg: str) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now_str}] {msg}"
    print(formatted, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def load_accounts() -> dict[str, dict[str, str]]:
    if not ACCOUNTS_FILE.exists():
        raise FileNotFoundError(f"Accounts file {ACCOUNTS_FILE} not found!")
    return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))


def get_kaggle_api(username: str):
    from kaggle.api.kaggle_api_extended import KaggleApi

    accounts = load_accounts()
    creds = accounts.get(username)
    if not creds:
        raise ValueError(f"No credentials found for account: {username}")

    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

    api = KaggleApi()
    api.authenticate()
    return api


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Warning: could not read {STATE_FILE}: {e}")

    # Initialize default state
    state = {
        "models": {m["id"]: m for m in DEFAULT_MODELS},
        "history": [],
    }
    save_state(state)
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def check_status(api, kernel_slug: str) -> tuple[str, str]:
    try:
        st = api.kernels_status(kernel_slug)
        raw_status = getattr(st, "status", st)
        status_str = getattr(raw_status, "name", str(raw_status))
        fail_msg = getattr(st, "failure_message", None) or getattr(st, "failureMessage", "") or ""
        return status_str, str(fail_msg)
    except Exception as e:
        return "ERROR", str(e)


def fetch_kernel_output_urls(api, owner: str, slug: str, max_retries: int = 4, delay: int = 15) -> dict[str, str]:
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

    for attempt in range(1, max_retries + 1):
        try:
            matched_urls: dict[str, str] = {}
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
                        url = getattr(f, "url", None)
                        if url:
                            matched_urls[fn] = url
                    token = getattr(res, "next_page_token", None)
                    if not token:
                        break

            if matched_urls:
                log(f"Found {len(matched_urls)} output files for {owner}/{slug}")
                return matched_urls

            log(f"Attempt {attempt}/{max_retries}: No output files yet for {owner}/{slug}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            log(f"Attempt {attempt}/{max_retries} error fetching output files for {owner}/{slug}: {e}")
            time.sleep(delay)

    return {}


def determine_last_epoch_from_log(log_url_or_text: str) -> int:
    """Parse log.txt content to find the last completed epoch."""
    lines = []
    if log_url_or_text.startswith("http://") or log_url_or_text.startswith("https://"):
        try:
            resp = requests.get(log_url_or_text, timeout=30)
            if resp.status_code == 200:
                lines = resp.text.splitlines()
        except Exception as e:
            log(f"Failed to fetch log from URL: {e}")
    else:
        lines = log_url_or_text.splitlines()

    max_epoch = -1
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Try JSON format: {"train_lr": ..., "epoch": 67, ...}
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "epoch" in data:
                    max_epoch = max(max_epoch, int(data["epoch"]))
                    return max_epoch
            except Exception:
                pass
        # Try text pattern: Epoch: [67] or Epoch [67/100]
        match = re.search(r"Epoch:\s*\[(\d+)\]", line)
        if match:
            return int(match.group(1))

    return max_epoch


def find_key_file_url(file_dict: dict[str, str], candidate_names: list[str]) -> str | None:
    for cand in candidate_names:
        for fname, url in file_dict.items():
            if fname.endswith(cand) or fname == cand:
                return url
    return None


def create_and_push_resume_kernel(model_info: dict[str, Any], urls: dict[str, str], last_epoch: int) -> str:
    resume_epoch = last_epoch + 1 if last_epoch >= 0 else 1
    log(f"Preparing resume kernel for {model_info['name']} starting from epoch {resume_epoch}...")

    # Find required URLs
    subdir = model_info.get("output_subdir", "output_dfine_m")
    last_pth_url = find_key_file_url(urls, [f"{subdir}/last.pth", "last.pth"])
    best_pth_url = find_key_file_url(urls, [f"{subdir}/best_stg1.pth", f"{subdir}/best_stg2.pth", "best_stg1.pth", "best.pth"])
    log_txt_url = find_key_file_url(urls, [f"{subdir}/log.txt", "log.txt"])

    if not last_pth_url:
        raise RuntimeError(f"Cannot resume {model_info['name']}: 'last.pth' URL not found in kernel output!")

    kernel_title = f"{model_info['slug_prefix']}{resume_epoch}"
    new_slug = f"{RESUME_ACCOUNT_NAME}/{kernel_title}"
    kernel_dir = KAGGLE_DIR / f"kernel_{model_info['type']}_resume{resume_epoch}_{RESUME_ACCOUNT_NAME}"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare code file
    template_path = Path(model_info["template_script"])
    if not template_path.exists():
        # Fallback to current directory code file
        candidates = list(KAGGLE_DIR.rglob(model_info["code_filename"]))
        if candidates:
            template_path = candidates[0]
        else:
            raise FileNotFoundError(f"Template script {template_path} not found!")

    code_content = template_path.read_text(encoding="utf-8")

    # Construct new FALLBACK_URLS block with strict 4-space function indentation
    fallback_dict = {
        "last.pth": last_pth_url,
    }
    if best_pth_url:
        fallback_dict["best_stg1.pth"] = best_pth_url
    if log_txt_url:
        fallback_dict["log.txt"] = log_txt_url

    url_lines = ["    FALLBACK_URLS = {"]
    for k, v in fallback_dict.items():
        url_lines.append(f'        "{k}": "{v}",')
    url_lines.append("    }")
    new_urls_block = "\n".join(url_lines)

    # Replace FALLBACK_URLS in code
    url_pattern = r"[ \t]*FALLBACK_URLS\s*=\s*\{.*?\}"
    if re.search(url_pattern, code_content, re.DOTALL):
        code_content = re.sub(url_pattern, new_urls_block, code_content, count=1, flags=re.DOTALL)
    else:
        log("Warning: could not find FALLBACK_URLS block regex. Inserting above resume_ckpt.")
        code_content = code_content.replace(
            'if (OUTPUT_DIR / "last.pth").exists():',
            f"{new_urls_block}\n    if (OUTPUT_DIR / \"last.pth\").exists():",
        )

    dest_code_file = kernel_dir / model_info["code_filename"]
    dest_code_file.write_text(code_content, encoding="utf-8")

    # 2. Prepare kernel-metadata.json
    dataset_source = model_info.get("dataset_source", "nguynvntnpht/scale-error-dc-v26-yolov8")
    meta = {
        "id": new_slug,
        "title": kernel_title,
        "code_file": model_info["code_filename"],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_source],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save URLs record locally
    (kernel_dir / "resume_urls.json").write_text(json.dumps(fallback_dict, indent=2), encoding="utf-8")

    # 3. Push kernel using target account
    log(f"Pushing resume kernel to Kaggle as {new_slug}...")
    api_resume = get_kaggle_api(RESUME_ACCOUNT_NAME)
    push_res = api_resume.kernels_push(str(kernel_dir))
    push_err = getattr(push_res, "error", None)
    if push_err:
        log(f"[ERROR] Failed to push kernel {new_slug}: {push_err}")
        raise RuntimeError(f"Failed to push kernel: {push_err}")

    log(f"[SUCCESS] Resume kernel launched: https://www.kaggle.com/code/{new_slug} (Epoch {resume_epoch} -> 100)")
    return new_slug


def download_artifacts(urls: dict[str, str], dest_folder: Path, model_info: dict[str, Any]) -> None:
    dest_folder.mkdir(parents=True, exist_ok=True)
    log(f"Downloading key artifacts to {dest_folder}...")

    patterns = [".pth", ".json", ".txt", ".csv"]
    for fn, url in urls.items():
        if any(fn.endswith(ext) for ext in patterns):
            if any(skip in fn for skip in ["data_roi", ".git", "coco", "dataset_split"]):
                continue
            dest_file = dest_folder / Path(fn).name
            if dest_file.exists() and dest_file.stat().st_size > 0:
                continue
            try:
                log(f"Downloading {Path(fn).name}...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(dest_file, "wb") as f:
                        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                log(f"Saved: {dest_file.name} ({dest_file.stat().st_size / (1024 * 1024):.1f} MB)")
            except Exception as e:
                log(f"Error downloading {fn}: {e}")

    # Copy best weight to destination
    candidates = list(dest_folder.glob("*best*.pth"))
    if candidates and "best_dest" in model_info:
        dest_pth = Path(model_info["best_dest"])
        shutil.copy2(candidates[0], dest_pth)
        log(f"Successfully updated best weights to {dest_pth}")


def check_and_process_model(model_info: dict[str, Any], state: dict[str, Any]) -> None:
    slug = model_info["current_slug"]
    owner, kernel_name = slug.split("/", maxsplit=1)
    api = get_kaggle_api(owner)

    status_str, fail_msg = check_status(api, slug)
    log(f"[{model_info['name']}] Kernel: {slug} | Status: {status_str} | Info: {fail_msg}")

    if status_str in {"RUNNING", "QUEUED", "KernelWorkerStatus.RUNNING", "KernelWorkerStatus.QUEUED"}:
        log(f"[{model_info['name']}] Still actively running. Next check in scheduled interval.")
        model_info["status"] = "RUNNING"
        return

    # If it is not running, session ended!
    log(f"[{model_info['name']}] Session is not running (Status: {status_str}). Analyzing output...")
    urls = fetch_kernel_output_urls(api, owner, kernel_name)

    # Determine last epoch
    last_epoch = -1
    log_url = find_key_file_url(urls, ["log.txt", "output_fbdnet_m/log.txt", "output_dfine_m/log.txt"])
    if log_url:
        last_epoch = determine_last_epoch_from_log(log_url)
        log(f"[{model_info['name']}] Identified last completed epoch: {last_epoch}")
    else:
        log(f"[{model_info['name']}] Warning: log.txt not found in outputs.")

    # Check if full training completed (100 epochs, index 99)
    if last_epoch >= 99 or (status_str in {"COMPLETE", "KernelWorkerStatus.COMPLETE"} and any("best_stg2" in f for f in urls)):
        log(f"[{model_info['name']}] TRAINING COMPLETED (100 Epochs)! Downloading final outputs...")
        save_dest = OUTPUT_DIR / kernel_name
        download_artifacts(urls, save_dest, model_info)
        model_info["status"] = "COMPLETE"
        state["history"].append({
            "model_id": model_info["id"],
            "slug": slug,
            "status": "COMPLETED",
            "epoch": last_epoch,
            "timestamp": datetime.now().isoformat(),
        })
        return

    # Otherwise session ended before reaching 100 epochs -> AUTO RESUME!
    log(f"[{model_info['name']}] Session ended at epoch {last_epoch} (target: 100). Initiating AUTO-RESUME...")
    try:
        new_slug = create_and_push_resume_kernel(model_info, urls, last_epoch)
        prev_slug = model_info["current_slug"]
        model_info["current_slug"] = new_slug
        model_info["status"] = "RUNNING"

        state["history"].append({
            "model_id": model_info["id"],
            "resumed_from": prev_slug,
            "new_slug": new_slug,
            "epoch": last_epoch,
            "timestamp": datetime.now().isoformat(),
        })
        log(f"[{model_info['name']}] State updated. Now tracking new kernel: {new_slug}")
    except Exception as e:
        log(f"[{model_info['name']}] Failed to auto-resume: {e}")
        model_info["status"] = f"RESUME_FAILED: {e}"


def run_monitoring_cycle(state: dict[str, Any]) -> None:
    log("=" * 70)
    log("Running Kaggle session status check cycle...")
    for model_id, model_info in list(state["models"].items()):
        if model_info.get("status") in {"COMPLETE", "CANCELLED"}:
            log(f"[{model_info['name']}] Status is {model_info.get('status')}. Skipping.")
            continue
        try:
            check_and_process_model(model_info, state)
        except Exception as e:
            log(f"Error checking model {model_id}: {e}")
        save_state(state)
    log("Check cycle completed.")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Auto-Resume Monitor")
    parser.add_argument("--interval", type=int, default=7200, help="Check interval in seconds (default: 7200s = 2h)")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    parser.add_argument("--status", action="store_true", help="Print current status and exit")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print("\n=== Current Kaggle Models Tracking Status ===")
        for mid, m in state["models"].items():
            print(f"- {m['name']} ({mid}):")
            print(f"    Current Slug: {m['current_slug']}")
            print(f"    Status:       {m.get('status', 'UNKNOWN')}")
        print("\nResume account:", RESUME_ACCOUNT_NAME)
        print("Log file:", LOG_FILE)
        return

    if args.once:
        run_monitoring_cycle(state)
        return

    log(f"Starting Kaggle Auto-Resume Monitor (Interval: {args.interval}s / {args.interval/3600:.1f}h)...")
    while True:
        try:
            run_monitoring_cycle(state)
        except Exception as e:
            log(f"Unexpected error in cycle: {e}")

        # Check if all completed
        all_done = all(m.get("status") == "COMPLETE" for m in state["models"].values())
        if all_done:
            log("ALL TRACKED MODELS HAVE COMPLETED TRAINING!")
            break

        log(f"Sleeping {args.interval} seconds until next check ({datetime.now().strftime('%H:%M:%S')} -> next check at {datetime.fromtimestamp(time.time() + args.interval).strftime('%H:%M:%S')})...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

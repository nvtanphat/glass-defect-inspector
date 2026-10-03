"""Download only durable artifacts from a large Kaggle kernel output."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path, PurePosixPath

import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHECKPOINT_NAMES = {
    "best_resume.pth",
    "last_resume.pth",
    "best_stg1.pth",
    "best_stg2.pth",
    "last.pth",
}
REPORT_NAMES = {
    "metrics_dfine_resume_test.json",
    "resume_summary.json",
    "log.txt",
}


def should_download(file_name: str) -> bool:
    path = PurePosixPath(file_name)
    output_parts = {"output_dfine_resume", "output_dfine_resume_final"}
    is_resume_output = any(part in output_parts for part in path.parts)
    if path.name in {"best_resume.pth", "last_resume.pth"}:
        return True
    if path.name in REPORT_NAMES:
        return is_resume_output or len(path.parts) == 1
    if re.fullmatch(r"checkpoint\d+\.pth", path.name):
        return is_resume_output
    return path.name in CHECKPOINT_NAMES and is_resume_output


def build_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def list_selected_files(
    api: KaggleApi,
    owner: str,
    slug: str,
    *,
    gcs_start_after: str | None = None,
    first_page_only: bool = False,
    page_size: int = 200,
):
    selected = []
    token = None
    if gcs_start_after:
        token = base64.b64encode(
            json.dumps(
                {"GcsPageToken": gcs_start_after}, separators=(",", ":")
            ).encode()
        ).decode()
    kernel_log = None
    page = 0
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, page_size, token)
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            page += 1
            files = response.files or []
            print(f"Scanned page {page}: {len(files)} files", flush=True)
            selected.extend(item for item in files if should_download(item.file_name))
            if kernel_log is None and response.log:
                kernel_log = response.log
            token = response.next_page_token
            if not token or first_page_only:
                break
    return selected, kernel_log


def remote_size(item) -> int:
    for field in ("total_bytes", "size", "file_size"):
        value = getattr(item, field, None)
        if value:
            return int(value)
    return 0


def download_with_resume(session: requests.Session, item, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    expected_size = remote_size(item)

    if output_path.exists() and expected_size and output_path.stat().st_size == expected_size:
        print(f"Already complete: {output_path}")
        return
    if output_path.exists() and output_path.stat().st_size == 0:
        output_path.unlink()

    for attempt in range(1, 21):
        downloaded = temporary_path.stat().st_size if temporary_path.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        print(
            f"Downloading {item.file_name}: attempt {attempt}, offset {downloaded:,}",
            flush=True,
        )
        try:
            with session.get(
                item.url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                append = downloaded > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                if downloaded and not append:
                    downloaded = 0

                content_range = response.headers.get("Content-Range", "")
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    expected_size = int(match.group(1))
                elif not expected_size and response.headers.get("Content-Length"):
                    expected_size = downloaded + int(response.headers["Content-Length"])

                next_progress = downloaded + 64 * 1024 * 1024
                with temporary_path.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            downloaded += len(chunk)
                            if downloaded >= next_progress:
                                print(f"  received {downloaded:,} bytes", flush=True)
                                next_progress += 64 * 1024 * 1024

            current_size = temporary_path.stat().st_size
            if expected_size and current_size < expected_size:
                raise OSError(f"Incomplete download: {current_size:,}/{expected_size:,} bytes")
            os.replace(temporary_path, output_path)
            print(f"Saved: {output_path} ({output_path.stat().st_size:,} bytes)", flush=True)
            return
        except (OSError, requests.RequestException) as error:
            print(f"Retryable download error: {error}", flush=True)
            if attempt == 20:
                raise
            time.sleep(min(30, attempt * 3))


def download_parallel_chunks(
    item,
    output_path: Path,
    *,
    workers: int = 6,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Download a large Kaggle output using independent resumable ranges."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe = requests.get(
        item.url,
        headers={"Range": "bytes=0-0"},
        timeout=(30, 90),
    )
    probe.raise_for_status()
    match = re.search(r"/(\d+)$", probe.headers.get("Content-Range", ""))
    if not match:
        raise RuntimeError("Kaggle download did not return the checkpoint size")
    total_size = int(match.group(1))
    if output_path.exists() and output_path.stat().st_size == total_size:
        print(f"Already complete: {output_path}", flush=True)
        return

    parts_dir = output_path.with_suffix(output_path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(start + chunk_size - 1, total_size - 1))
        for index, start in enumerate(range(0, total_size, chunk_size))
    ]

    def fetch(spec: tuple[int, int, int]) -> Path:
        index, start, end = spec
        part = parts_dir / f"{index:05d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return part
        for attempt in range(1, 11):
            try:
                with requests.get(
                    item.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    with part.open("wb") as output:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                output.write(block)
                if part.stat().st_size != expected:
                    raise OSError(
                        f"chunk {index} incomplete: {part.stat().st_size}/{expected}"
                    )
                print(
                    f"Chunk {index + 1}/{len(ranges)} downloaded "
                    f"({end + 1:,}/{total_size:,})",
                    flush=True,
                )
                return part
            except (OSError, requests.RequestException) as error:
                if attempt == 10:
                    raise
                print(f"Chunk {index + 1} retry {attempt}: {error}", flush=True)
                time.sleep(min(20, attempt * 2))
        raise AssertionError("unreachable")

    print(
        f"Downloading {item.file_name}: {total_size:,} bytes in "
        f"{len(ranges)} chunks with {workers} workers",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, ranges))

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    with temporary_path.open("wb") as output:
        for index, _, _ in ranges:
            part = parts_dir / f"{index:05d}.part"
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if temporary_path.stat().st_size != total_size:
        raise OSError("Merged checkpoint size does not match the remote file")
    os.replace(temporary_path, output_path)
    for part in parts_dir.iterdir():
        part.unlink()
    parts_dir.rmdir()
    print(f"Saved: {output_path} ({total_size:,} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--gcs-start-after")
    parser.add_argument("--first-page-only", action="store_true")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--parallel-chunks", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    args = parser.parse_args()

    owner, slug = args.kernel.split("/", maxsplit=1)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    selected, log = list_selected_files(
        api,
        owner,
        slug,
        gcs_start_after=args.gcs_start_after,
        first_page_only=args.first_page_only,
        page_size=args.page_size,
    )
    print(f"Selected durable artifacts: {len(selected)}")
    for item in selected:
        print(f"  - {item.file_name}")

    if log:
        log_path = destination / f"{slug}.log"
        log_path.write_text(log, encoding="utf-8")
        print(f"Saved log: {log_path}")

    session = build_session()
    for item in selected:
        relative = Path(*PurePosixPath(item.file_name).parts)
        output_path = destination / relative
        if args.parallel_chunks:
            download_parallel_chunks(
                item,
                output_path,
                workers=args.workers,
                chunk_size=args.chunk_size_mb * 1024 * 1024,
            )
        else:
            download_with_resume(session, item, output_path)


if __name__ == "__main__":
    main()

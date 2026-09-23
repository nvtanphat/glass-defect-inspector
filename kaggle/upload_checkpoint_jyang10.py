import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "kaggle" / "checkpoint_upload_jyang10"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

os.environ["KAGGLE_USERNAME"] = "jyang10"
os.environ["KAGGLE_KEY"] = "083346fbb9da8ad61b964e678d18a71e"

def upload_checkpoint():
    src_pt = ROOT / "kaggle_eval_results" / "best.pt"
    dst_pt = UPLOAD_DIR / "best.pt"
    if not dst_pt.exists() or dst_pt.stat().st_size != src_pt.stat().st_size:
        print(f"Copying {src_pt} -> {dst_pt}...")
        shutil.copy2(src_pt, dst_pt)

    meta_path = UPLOAD_DIR / "dataset-metadata.json"
    meta_content = '''{
  "title": "scale-error-yolo26 checkpoint",
  "id": "jyang10/scale-error-yolo26-checkpoint",
  "licenses": [{"name": "CC0-1.0"}]
}'''
    meta_path.write_text(meta_content, encoding="utf-8")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        print("Creating dataset jyang10/scale-error-yolo26-checkpoint...")
        api.dataset_create_new(
            folder=str(UPLOAD_DIR),
            public=False,
            quiet=False,
        )
        print("Checkpoint dataset created successfully!")
    except Exception as e:
        print(f"Create failed ({e}), creating version...")
        api.dataset_create_version(
            folder=str(UPLOAD_DIR),
            version_notes="Upload verified best.pt (39.0% mAP)",
        )
        print("Checkpoint version updated successfully!")

if __name__ == "__main__":
    upload_checkpoint()

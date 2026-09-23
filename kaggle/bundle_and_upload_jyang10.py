import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "kaggle" / "code_bundle_upload_jyang10"

os.environ["KAGGLE_USERNAME"] = "jyang10"
os.environ["KAGGLE_KEY"] = "083346fbb9da8ad61b964e678d18a71e"

def build_zip():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = UPLOAD_DIR / "code_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating zip at {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for f_name in ["pyproject.toml", "README.md"]:
            p = ROOT / f_name
            if p.exists():
                zf.write(p, arcname=f_name)
                shutil.copy2(p, UPLOAD_DIR / f_name)

        # Add configs
        for p in (ROOT / "configs").glob("*.yaml"):
            zf.write(p, arcname=f"configs/{p.name}")

        # Add processing
        for p in (ROOT / "processing").glob("*.py"):
            zf.write(p, arcname=f"processing/{p.name}")
        for p in (ROOT / "processing").glob("*.json"):
            zf.write(p, arcname=f"processing/{p.name}")

        # Add src
        for p in (ROOT / "src").rglob("*.py"):
            rel = p.relative_to(ROOT)
            zf.write(p, arcname=str(rel).replace("\\", "/"))

    print(f"Zip created, size: {zip_path.stat().st_size / 1024:.1f} KB")

    meta_path = UPLOAD_DIR / "dataset-metadata.json"
    meta_content = '''{
  "title": "scale-error-yolo26 code",
  "id": "jyang10/scale-error-yolo26-code",
  "licenses": [{"name": "CC0-1.0"}]
}'''
    meta_path.write_text(meta_content, encoding="utf-8")

def create_or_update_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        print("Attempting to create dataset jyang10/scale-error-yolo26-code...")
        api.dataset_create_new(
            folder=str(UPLOAD_DIR),
            dir_mode="zip",
            public=False,
            quiet=False,
        )
        print("Dataset created successfully!")
    except Exception as e:
        print(f"Create failed ({e}), attempting version update...")
        api.dataset_create_version(
            folder=str(UPLOAD_DIR),
            version_notes="Upload code bundle for jyang10",
            dir_mode="zip",
        )
        print("Dataset version updated successfully!")

if __name__ == "__main__":
    build_zip()
    create_or_update_dataset()

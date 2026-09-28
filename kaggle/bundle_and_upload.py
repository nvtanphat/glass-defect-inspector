import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "kaggle" / "code_bundle_upload"

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
                # also copy to UPLOAD_DIR
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
    
    # Check contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        print("Zip contents:")
        for name in sorted(zf.namelist()):
            print("  ", name)

    # Ensure dataset-metadata.json in UPLOAD_DIR
    meta_path = UPLOAD_DIR / "dataset-metadata.json"
    meta_content = '''{
  "title": "scale-error-yolo26 code",
  "id": "nguynvntnpht/scale-error-yolo26-code",
  "licenses": [{"name": "CC0-1.0"}]
}'''
    meta_path.write_text(meta_content, encoding="utf-8")

def upload_dataset():
    print("Uploading updated code dataset to Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_version(
        folder=str(UPLOAD_DIR),
        version_notes="D-FINE T4-safe batch 4, reduced augmentation, 30-epoch refinement",
        dir_mode="zip",
    )
    print("Code dataset version updated successfully!")

if __name__ == "__main__":
    build_zip()
    upload_dataset()

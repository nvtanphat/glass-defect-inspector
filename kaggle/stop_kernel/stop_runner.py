from pathlib import Path
import json
import subprocess
import os

stop_dir = Path("kaggle/stop_kernel")
stop_dir.mkdir(parents=True, exist_ok=True)
(stop_dir / "stop.py").write_text("print('Kernel stopped successfully.')\n", encoding="utf-8")
meta = {
    "id": "leppan/scale-error-dfine-m-resume-71-100",
    "title": "scale-error-dfine-m-resume-71-100",
    "code_file": "stop.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
}
(stop_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

os.environ["KAGGLE_USERNAME"] = "leppan"
os.environ["KAGGLE_KEY"] = "3ca126b0eff148b5a3d8647ad4688625"
res = subprocess.run(["kaggle", "kernels", "push", "-p", str(stop_dir)], capture_output=True, text=True)
print("Push stdout:", res.stdout)
print("Push stderr:", res.stderr)

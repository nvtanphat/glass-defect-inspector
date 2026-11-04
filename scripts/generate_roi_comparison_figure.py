"""Generate Authentic Before & After ROI Figure (Gọn, sạch, ảnh thật từ dataset).

Visual Comparison:
- Panel (a): Trước khi cắt ROI (Ảnh gốc: 5328 × 3000, bao gồm viền băng chuyền và khung cắt ROI)
- Panel (b): Sau khi cắt ROI (Vùng phôi kính: 3980 × 3000, loại bỏ hoàn toàn biên băng chuyền thừa)

Standards:
- Sử dụng 100% ảnh chụp camera thực tế từ dataset
- Không vẽ biểu đồ, không vẽ hình nhân tạo, không làm rối ảnh
- Background trắng thuần khiết (#FFFFFF)
- Chuẩn 300 DPI, không gian màu RGB
- Xuất trực tiếp ra docs/assets/roi_vs_no_roi_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS_ASSETS = ROOT / "docs" / "assets"
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "processing" / "roi_config.json"
EDA_PATH = ROOT / "artifacts" / "eda_review" / "cell_13_0.png"

# Publication Typography
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["text.color"] = "#0F172A"


def load_real_dataset_image() -> np.ndarray:
    """Load authentic raw camera frame from real dataset captures."""
    if not EDA_PATH.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {EDA_PATH}")
    eda_img = cv2.imread(str(EDA_PATH))
    if eda_img is None:
        raise ValueError(f"Could not decode image at {EDA_PATH}")

    # Row 1 corresponds to a real 5328x3000 production glass inspection frame
    raw_bgr = eda_img[614:1006, 12:700]
    return raw_bgr


def generate_real_roi_figure(out_filename: str = "roi_vs_no_roi_comparison.png") -> Path:
    raw_bgr = load_real_dataset_image()
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = raw_rgb.shape

    # Read ROI boundaries from roi_config.json
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        roi_xyxyn = config.get("groups", {}).get("5328x3000", {}).get("roi_xyxyn", [0.14113, 0.0, 0.88813, 1.0])
    else:
        roi_xyxyn = [0.14113178913288288, 0.0, 0.8881345303115614, 1.0]

    x_left_frac, _, x_right_frac, _ = roi_xyxyn
    rx_left = int(round(w * x_left_frac))
    rx_right = int(round(w * x_right_frac))
    w_crop = rx_right - rx_left

    cropped_rgb = raw_rgb[:, rx_left:rx_right]

    # Side-by-side with proportional width ratios so vertical scales align 1:1
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [w, w_crop]},
        facecolor="#FFFFFF",
    )

    # -------------------------------------------------------------------------
    # Panel (a): Trước khi cắt ROI
    # -------------------------------------------------------------------------
    ax1.set_facecolor("#FFFFFF")
    ax1.imshow(raw_rgb)

    # Clean ROI bounding rectangle indicating crop boundaries
    roi_rect = patches.Rectangle(
        (rx_left, 0),
        w_crop,
        h - 1,
        linewidth=2.0,
        edgecolor="#DC2626",
        facecolor="none",
        linestyle="-",
    )
    ax1.add_patch(roi_rect)
    ax1.set_title(
        "(a) Trước khi cắt ROI (Ảnh gốc: 5328 × 3000)",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax1.axis("off")

    # -------------------------------------------------------------------------
    # Panel (b): Sau khi cắt ROI
    # -------------------------------------------------------------------------
    ax2.set_facecolor("#FFFFFF")
    ax2.imshow(cropped_rgb)

    # Clean framing border
    border_rect = patches.Rectangle(
        (0, 0),
        w_crop - 1,
        h - 1,
        linewidth=1.2,
        edgecolor="#94A3B8",
        facecolor="none",
        linestyle="-",
    )
    ax2.add_patch(border_rect)
    ax2.set_title(
        "(b) Sau khi cắt ROI (Vùng phôi kính: 3980 × 3000)",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax2.axis("off")

    plt.tight_layout(w_pad=2.5)

    out_path = DOCS_ASSETS / out_filename
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)

    # Ensure pure RGB color space (no alpha channel)
    with Image.open(out_path) as im:
        if im.mode != "RGB":
            im.convert("RGB").save(str(out_path), quality=100)

    print(f"Successfully generated authentic ROI comparison figure: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_real_roi_figure()

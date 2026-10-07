"""Script to visualize and compare confusion matrices across models."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
LABELS = CLASSES + ["background"]

# 1. YOLO26m Normalized Confusion Matrix (from report_assets/yolo26m_cm.png)
cm_yolo26m = np.array([
    [0.49, 0.01, 0.00, 0.00, 0.00, 0.14],
    [0.01, 0.57, 0.00, 0.01, 0.00, 0.27],
    [0.00, 0.01, 0.88, 0.21, 0.07, 0.45],
    [0.00, 0.00, 0.02, 0.65, 0.00, 0.04],
    [0.00, 0.00, 0.00, 0.00, 0.67, 0.10],
    [0.50, 0.41, 0.10, 0.13, 0.25, 0.00]
])

# 2. YOLO11m Normalized Confusion Matrix (from report_assets/yolo11m_cm.png)
cm_yolo11m = np.array([
    [0.32, 0.01, 0.00, 0.00, 0.00, 0.12],
    [0.02, 0.48, 0.00, 0.01, 0.00, 0.26],
    [0.00, 0.01, 0.88, 0.20, 0.02, 0.54],
    [0.00, 0.00, 0.02, 0.71, 0.05, 0.05],
    [0.00, 0.00, 0.00, 0.00, 0.72, 0.04],
    [0.66, 0.50, 0.10, 0.08, 0.20, 0.00]
])

def plot_cm(mat, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("True (Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "#08306b"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=color, fontweight="semibold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized Proportion")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    os.makedirs("report_assets", exist_ok=True)
    plot_cm(cm_yolo26m, "YOLO26m Normalized Confusion Matrix (Test Split)", "report_assets/yolo26m_cm.png")
    plot_cm(cm_yolo11m, "YOLO11m Normalized Confusion Matrix (Test Split)", "report_assets/yolo11m_cm.png")
    print("Confusion matrices generated successfully.")

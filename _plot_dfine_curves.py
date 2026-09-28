"""Ve curve train cho D-FINE-M:
  1) dfine_m_loss_curve.png  - loss + lr theo step (tu tfevents)
  2) dfine_m_map_curve.png   - mAP/AR theo epoch (tu log.txt + tfevents)
"""
import json
import struct
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.compat.proto import event_pb2

TF_EVENTS = "results_report/output_dfine_m/summary/events.out.tfevents.1790518295.c8529f1bcf2a.54.0"
LOG_TXT = "results_report/output_dfine_m/log.txt"
OUT_DIR = "report_assets"


def read_records(path):
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                return
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            if len(data) < length:
                return
            f.read(4)
            yield data


def scalar(v):
    if v.HasField("simple_value"):
        return v.simple_value
    t = v.tensor
    if t.dtype == 1:
        if t.float_val:
            return t.float_val[0]
        if t.tensor_content:
            return struct.unpack("<f", t.tensor_content[:4])[0]
    return None


def load_tfevents(path):
    tags = defaultdict(list)
    for data in read_records(path):
        ev = event_pb2.Event.FromString(data)
        for v in ev.summary.value:
            s = scalar(v)
            if s is not None:
                tags[v.tag].append((ev.step, s))
    return tags


def ema(points, win):
    """Chay trung binh cong de lam mem duong loss."""
    win = int(win)
    out, acc = [], 0.0
    for i, (_, v) in enumerate(points):
        acc += v
        if i + 1 >= win:
            if i + 1 > win:
                acc -= points[i + 1 - win][1]
            out.append((points[i][0], acc / win))
    return out


def main():
    tags = load_tfevents(TF_EVENTS)
    rows = [json.loads(l) for l in open(LOG_TXT, encoding="utf-8") if l.strip().startswith("{")]

    # ---------- 1) LOSS + LR ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=130)

    step0 = tags["Loss/total"][0][0]
    n_steps = tags["Loss/total"][-1][0] - step0

    # total loss (truc trai)
    pts = ema(tags["Loss/total"], 40)
    ax1.plot([p[0] - step0 for p in pts], [p[1] for p in pts], lw=1.7,
             label="Total loss", color="#e63946")
    ax1.set_ylabel("total loss", color="#e63946", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#e63946")

    # cac thanh phan loss (truc phai vi total ~18 de bep het)
    ax1b = ax1.twinx()
    for (tag, label), color in zip([
        ("Loss/loss_fgl", "loss_fgl (fine-grained)"),
        ("Loss/loss_vfl", "loss_vfl (Varifocal)"),
        ("Loss/loss_giou", "loss_giou"),
        ("Loss/loss_bbox", "loss_bbox"),
    ], ["#457b9d", "#ff9f1c", "#2a9d8f", "#9b5de5"]):
        pts = ema(tags[tag], 40)
        ax1b.plot([p[0] - step0 for p in pts], [p[1] for p in pts], lw=1.6, label=label, color=color)
    ax1b.set_ylabel("loss thành phần", fontsize=10)
    ax1b.set_yscale("log")
    ax1b.set_ylim(0.01, 3)

    ax1.set_title("D-FINE-M · Stage 2 (epoch 88→99): loss theo step\n"
                  "(EMA smoothing cửa sổ 40 · 8.400 step · log mỗi 10 step)", fontsize=11)
    ax1.set_xlabel("step")
    ax1.set_xlim(0, n_steps)
    ax1.grid(alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)

    for tag, label in [("Lr/pg_0", "lr backbone (pg_0/pg_1)"), ("Lr/pg_2", "lr head (pg_2/pg_3)")]:
        pts = tags[tag]
        ax2.plot([p[0] - step0 for p in pts], [p[1] for p in pts], lw=1.8, label=label)
    ax2.set_title("Learning rate stage 2 (cosine ở mức nền 2.5e-5)", fontsize=11)
    ax2.set_xlabel("step")
    ax2.set_ylabel("lr")
    ax2.set_xlim(0, n_steps)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dfine_m_loss_curve.png", bbox_inches="tight")
    print("DA LUU", f"{OUT_DIR}/dfine_m_loss_curve.png")

    # ---------- 2) mAP THEO EPOCH ----------
    epochs = [r["epoch"] for r in rows]
    ap = [r["test_coco_eval_bbox"][0] * 100 for r in rows]
    ap50 = [r["test_coco_eval_bbox"][1] * 100 for r in rows]
    ap75 = [r["test_coco_eval_bbox"][2] * 100 for r in rows]
    aps = [r["test_coco_eval_bbox"][3] * 100 for r in rows]
    apm = [r["test_coco_eval_bbox"][4] * 100 for r in rows]
    apl = [r["test_coco_eval_bbox"][5] * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=130)

    ax1.plot(epochs, ap50, "o-", lw=2, label="mAP@50", color="#2a9d8f")
    ax1.plot(epochs, ap, "o-", lw=2.4, label="mAP@50:95", color="#e63946")
    ax1.plot(epochs, ap75, "o-", lw=2, label="mAP@75", color="#457b9d")
    best = max(range(len(ap)), key=lambda i: ap[i])
    ax1.annotate(f"best mAP@50:95 = {ap[best]:.2f}%\n(epoch {epochs[best]})",
                 (epochs[best], ap[best]), textcoords="offset points", xytext=(10, -26),
                 fontsize=9, color="#e63946",
                 arrowprops=dict(arrowstyle="->", color="#e63946"))
    ax1.set_title("D-FINE-M · mAP trên test set (890 ảnh) theo epoch", fontsize=11)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("AP (%)")
    ax1.set_xticks(epochs)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 100)

    ax2.plot(epochs, apl, "s-", lw=2, label="AP large")
    ax2.plot(epochs, apm, "s-", lw=2, label="AP medium")
    ax2.plot(epochs, aps, "s-", lw=2, label="AP small")
    ax2.set_title("mAP theo kích thước đối tượng (small bị thiệt nhất)", fontsize=11)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("AP (%)")
    ax2.set_xticks(epochs)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 70)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dfine_m_map_curve.png", bbox_inches="tight")
    print("DA LUU", f"{OUT_DIR}/dfine_m_map_curve.png")

    print("\nSo lieu mAP (epoch cuoi %d): mAP50:95=%.2f  mAP50=%.2f  mAP75=%.2f" %
          (epochs[-1], ap[-1], ap50[-1], ap75[-1]))
    print("Best epoch %d: mAP50:95=%.2f" % (epochs[best], ap[best]))


if __name__ == "__main__":
    main()

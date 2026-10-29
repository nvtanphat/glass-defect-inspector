"""Grad-CAM visualization for YOLO26 defect detection models."""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


class YOLOGradCAM:
    def __init__(self, model_path="weights/yolo26m.pt", target_layer_idx=16):
        self.model_path = model_path
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.names = self.yolo.names
        self.target_layer_idx = target_layer_idx

        self.activations = []
        self.gradients = []
        self._hooks_registered = False

    def _register_hooks(self):
        if not self._hooks_registered:
            target_layer = self.model.model[self.target_layer_idx]
            target_layer.register_forward_hook(self._forward_hook)
            target_layer.register_full_backward_hook(self._backward_hook)
            self._hooks_registered = True

    def _forward_hook(self, module, inp, out):
        self.activations.append(out)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def generate(self, img_path, conf_thresh=0.25, imgsz=960):
        self.activations.clear()
        self.gradients.clear()

        # Load and preprocess image
        orig_img = cv2.imread(str(img_path))
        orig_h, orig_w = orig_img.shape[:2]

        # Resize/pad to model size (letterbox style or standard resize)
        input_img = cv2.resize(orig_img, (imgsz, imgsz))
        rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_img.requires_grad = True

        # Run detection to get boxes
        results = self.yolo.predict(img_path, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
        boxes = results.boxes

        # Register hooks for Grad-CAM
        self._register_hooks()

        cam_results = []

        if len(boxes) == 0:
            # If no boxes passed threshold, take highest activation overall
            self.model.zero_grad()
            self.activations.clear()
            self.gradients.clear()
            out = self.model(tensor_img)
            pred = out[0]

            class_scores = pred[0, 4:, :]
            top_score, top_idx = class_scores.max(dim=1)
            target_class = int(top_score.argmax())
            score = class_scores[target_class].max()
            score.backward(retain_graph=True)

            act = self.activations[-1][0]  # [C, H, W]
            grad = self.gradients[-1][0]  # [C, H, W]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((weights * act).sum(dim=0)).detach().cpu().numpy()
            cam = cv2.resize(cam, (orig_w, orig_h))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            cam_results.append(
                {
                    "class_id": target_class,
                    "class_name": self.names[target_class],
                    "conf": float(score.detach().cpu().numpy()),
                    "box": None,
                    "cam": cam,
                }
            )
        else:
            for b in boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())
                xyxy = b.xyxy[0].cpu().numpy()

                # Backpropagate score of the target class
                self.model.zero_grad()
                self.activations.clear()
                self.gradients.clear()

                out = self.model(tensor_img)
                pred = out[0]
                target_score = pred[0, 4 + cls_id, :].max()
                target_score.backward(retain_graph=True)

                act = self.activations[-1][0]
                grad = self.gradients[-1][0]
                weights = grad.mean(dim=(1, 2), keepdim=True)
                cam = F.relu((weights * act).sum(dim=0)).detach().cpu().numpy()
                cam = cv2.resize(cam, (orig_w, orig_h))
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                cam_results.append(
                    {
                        "class_id": cls_id,
                        "class_name": self.names[cls_id],
                        "conf": conf,
                        "box": xyxy,
                        "cam": cam,
                    }
                )
                break  # Visualize top detection

        return orig_img, cam_results


def create_visualization_figure(orig_img, cam_result, output_path, title_prefix="Grad-CAM"):
    h, w = orig_img.shape[:2]
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    cam = cam_result["cam"]
    cls_name = cam_result["class_name"]
    conf = cam_result["conf"]
    box = cam_result["box"]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap_rgb) / 255.0 * 0.5 + np.float32(rgb_img) / 255.0 * 0.5
    overlay = np.clip(overlay, 0, 1)

    # Image with box
    img_with_box = rgb_img.copy()
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 50, 50), 3)
        cv2.putText(
            img_with_box,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 50, 50),
            2,
        )

        # Also draw box on overlay
        overlay_uint8 = np.uint8(overlay * 255)
        cv2.rectangle(overlay_uint8, (x1, y1), (x2, y2), (255, 255, 255), 3)
        overlay = np.float32(overlay_uint8) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    axes[0].imshow(img_with_box)
    axes[0].set_title(
        f"Ảnh gốc & Detection ({cls_name}: {conf:.2f})", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap (Feature Map C3k2 P3)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Overlay (Vùng kích hoạt hội tụ vào khuyết tật)", fontsize=11, fontweight="bold"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"{title_prefix} - Phát hiện lỗi miệng chai thủy tinh ({cls_name})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Da luu anh Grad-CAM tai: {output_path}")


def run_all_samples():
    cam_engine = YOLOGradCAM(model_path="weights/yolo26m.pt", target_layer_idx=16)
    sample_files = [
        ("src/glass_inspector/sample_images/sample_bottle_1.jpg", "DiVat"),
        ("src/glass_inspector/sample_images/sample_bottle_2.jpg", "LoiChi"),
        ("src/glass_inspector/sample_images/sample_bottle_3.jpg", "DiVat"),
    ]

    saved_paths = []
    summary_data = []

    for i, (sample_file, expected_defect) in enumerate(sample_files, start=1):
        if not os.path.exists(sample_file):
            print(f"Bo qua: {sample_file} khong ton tai")
            continue

        orig_img, cam_results = cam_engine.generate(sample_file, conf_thresh=0.25)
        out_path = f"docs/assets/gradcam_sample_{i}_{expected_defect}.png"
        create_visualization_figure(orig_img, cam_results[0], out_path, title_prefix=f"Mẫu {i}")
        saved_paths.append(out_path)
        summary_data.append(
            (
                orig_img,
                cam_results[0],
                f"Mẫu {i}: {cam_results[0]['class_name']} ({cam_results[0]['conf']:.2f})",
            )
        )

    # Create combined summary grid (3 rows x 3 columns)
    if len(summary_data) > 0:
        fig, axes = plt.subplots(
            len(summary_data), 3, figsize=(14, 4.5 * len(summary_data)), dpi=140
        )
        if len(summary_data) == 1:
            axes = np.expand_dims(axes, 0)

        for row, (img, res, label) in enumerate(summary_data):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam = res["cam"]
            box = res["box"]
            cls_name = res["class_name"]
            conf = res["conf"]

            # 1. Image with box
            im_box = rgb.copy()
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(im_box, (x1, y1), (x2, y2), (255, 50, 50), 3)
                cv2.putText(
                    im_box,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 50, 50),
                    2,
                )

            # 2. Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 3. Overlay
            overlay = np.float32(heatmap_rgb) / 255.0 * 0.5 + np.float32(rgb) / 255.0 * 0.5
            overlay = np.clip(overlay, 0, 1)
            if box is not None:
                overlay_u8 = np.uint8(overlay * 255)
                cv2.rectangle(overlay_u8, (x1, y1), (x2, y2), (255, 255, 255), 3)
                overlay = np.float32(overlay_u8) / 255.0

            axes[row, 0].imshow(im_box)
            axes[row, 0].set_title(f"Ảnh thực tế: {label}", fontsize=11, fontweight="bold")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(cam, cmap="jet")
            axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title(
                "Overlay tập trung vùng khuyết tật", fontsize=11, fontweight="bold"
            )
            axes[row, 2].axis("off")

        plt.suptitle(
            "BẢNG MINH HỌA GRAD-CAM PHÁT HIỆN LỖI MIỆNG CHAI THỦY TINH (YOLO26m)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        summary_out = "docs/assets/gradcam_summary.png"
        plt.savefig(summary_out, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Da luu tong hop Grad-CAM tai: {summary_out}")


if __name__ == "__main__":
    run_all_samples()

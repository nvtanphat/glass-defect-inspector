"""Hàm mất mát phân lớp Varifocal Loss (VFL) kết hợp trọng số lớp nghịch đảo tần suất.

Hàm mất mát mặc định `v8DetectionLoss` của Ultralytics dùng `BCEWithLogitsLoss` đối xứng.
Trong tập dữ liệu này, phân bố lớp cực kỳ mất cân bằng (LoiChi: 5282 hộp vs LoiTray: 249 hộp)
và 59% số hộp có diện tích dưới 0.25% diện tích ảnh.
`VFLDetectionLoss` thay thế nhánh phân loại bằng Varifocal Loss (Zhang et al.),
sử dụng target_scores từ TaskAlignedAssigner làm nhãn mềm IoU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics.utils.loss import v8DetectionLoss


class VFLDetectionLoss(v8DetectionLoss):
    """Hàm mất mát v8DetectionLoss nâng cấp với Varifocal Loss và trọng số phân lớp."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        gamma: float = 2.0,
        alpha: float = 0.75,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.gamma = gamma
        self.alpha = alpha
        cw = getattr(model, "class_weights", None)
        if cw is not None:
            self.class_weights = cw

    def get_assigned_targets_and_loss(self, preds, batch):
        """Phân bổ mục tiêu (Task-Aligned Assigner) và tính toán tổng tổn thất (Box + VFL + DFL)."""
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Gán nhãn mục tiêu Task-Aligned Assigner (TAL)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Varifocal classification loss: target_scores là điểm IoU mềm
        target_scores_typed = target_scores.to(dtype)
        one_hot = (target_scores_typed > 0).to(dtype)

        # Tính toán Varifocal Loss theo từng phần tử
        vfl_weight = (
            self.alpha * pred_scores.sigmoid().pow(self.gamma) * (1 - one_hot)
            + target_scores_typed * one_hot
        )
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_scores.float(), target_scores_typed.float(), reduction="none"
        )
        cls_loss_elements = (bce * vfl_weight).to(dtype)

        # Áp dụng trọng số phân lớp nếu có
        if getattr(self, "class_weights", None) is not None:
            cw = self.class_weights.to(device=cls_loss_elements.device, dtype=dtype)
            if cw.ndim == 1:
                cw = cw.view(1, 1, -1)
            cls_loss_elements = cls_loss_elements * cw

        cls_loss = cls_loss_elements.sum()
        loss[1] = cls_loss / target_scores_sum

        # Tổn thất tọa độ bounding box (CIoU Loss và Distribution Focal Loss)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        else:
            loss[0] += pred_distri[..., :0].sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach(), strict=True)),
        )


def compute_class_weights(dataset_report_path: str | Path, split: str = "train") -> torch.Tensor:
    """Tính trọng số nghịch đảo tần suất cho từng phân lớp từ file dataset_report.json thực tế.

    weight_c = mean(count) / count_c, giúp trọng số trung bình toàn cục bằng 1.0
    và phân lớp thiểu số (LoiTray, LoiNhua) nhận trọng số mất mát cao hơn.
    """
    report = json.loads(Path(dataset_report_path).read_text(encoding="utf-8"))
    if "names" in report:
        classes = [report["names"][str(i)] for i in range(len(report["names"]))]
    else:
        classes = report["classes"]
    counts = report["splits"][split]["class_counts"]
    ordered_counts = torch.tensor([float(counts[c]) for c in classes])
    return ordered_counts.mean() / ordered_counts

"""Các hàm mất mát và cân bằng phân lớp phục vụ chưng cất tri thức (Knowledge Distillation).

Bao gồm:
- compute_smooth_class_weights: Tính trọng số phân lớp làm mịn (căn bậc hai nghịch đảo tần số)
  nhằm chống mất cân bằng lớp mà không gây nổ gradient.
- make_vfl_loss_elements: Cài đặt tổn thất Varifocal Loss (VFL) với trọng số lớp theo từng pixel/anchor.
- ConfidenceWeightedVFLLoss: Lớp bọc tích hợp trực tiếp vào Ultralytics YOLO26/v8 criterion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_smooth_class_weights(
    class_counts: dict[int, int] | list[int],
    num_classes: int = 5,
    power: float = 0.5,
    min_weight: float = 0.6,
    max_weight: float = 2.5,
) -> torch.Tensor:
    """Tính toán trọng số cân bằng lớp làm mịn dựa trên căn bậc hai tần suất hiệu dụng.

    Tránh việc dùng nghịch đảo tuyến tính cực đoan (ví dụ: chênh lệch 17 lần giữa LoiChi và LoiTray)
    làm mất ổn định luồng gradient trong quá trình chưng cất tri thức.

    Tham số:
        class_counts: Số lượng hộp theo từng ID phân lớp.
        num_classes: Tổng số phân lớp (5 lớp).
        power: Số mũ làm mịn (0.5 tương đương căn bậc hai, 1.0 tương đương nghịch đảo tuyệt đối).
        min_weight: Ngưỡng chặn dưới của trọng số (mặc định: 0.6).
        max_weight: Ngưỡng chặn trên của trọng số (mặc định: 2.5).

    Trả về:
        torch.Tensor kích thước (num_classes,) chứa trọng số phân lớp đã chuẩn hóa về trung bình 1.0.
    """
    counts = [0] * num_classes
    if isinstance(class_counts, dict):
        for k, v in class_counts.items():
            if 0 <= int(k) < num_classes:
                counts[int(k)] = max(int(v), 1)
    else:
        for idx, v in enumerate(class_counts):
            if idx < num_classes:
                counts[idx] = max(int(v), 1)

    # Tính toán tỷ lệ căn bậc hai tham chiếu theo trung vị (median)
    med = float(np.median(counts))
    raw_weights = [(med / c) ** power for c in counts]

    # Chuẩn hóa sao cho trọng số trung bình toàn cục bằng 1.0
    mean_w = float(np.mean(raw_weights))
    norm_weights = [w / max(mean_w, 1e-6) for w in raw_weights]

    # Kẹp giá trị trong khoảng an toàn để tránh nổ hoặc tiêu biến gradient
    clamped_weights = [min(max(w, min_weight), max_weight) for w in norm_weights]
    return torch.tensor(clamped_weights, dtype=torch.float32)


def make_vfl_loss_elements(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tính toán các thành phần tổn thất Varifocal Loss (VFL) kèm trọng số lớp tùy chọn.

    Định nghĩa toán học của Varifocal Loss:
        VFL(p, q) =
            - q * log(p)                           nếu q > 0 (vùng tiền cảnh, q là điểm IoU mục tiêu)
            - alpha * p^gamma * log(1 - p)         nếu q == 0 (vùng hậu cảnh, phạt âm tính giả)
    """
    dtype = pred_scores.dtype
    target_scores_typed = target_scores.to(dtype)
    one_hot = (target_scores_typed > 0).to(dtype)

    p = pred_scores.sigmoid()
    vfl_weight = alpha * p.pow(gamma) * (1.0 - one_hot) + target_scores_typed * one_hot
    bce = F.binary_cross_entropy_with_logits(
        pred_scores.float(), target_scores_typed.float(), reduction="none"
    ).to(dtype)

    loss_elements = bce * vfl_weight

    # Nhân trọng số phân lớp nếu có chỉ định
    if class_weights is not None:
        cw_dev = class_weights.to(device=loss_elements.device, dtype=dtype)
        if cw_dev.ndim == 1:
            nc = cw_dev.shape[0]
            if loss_elements.ndim == 3:
                if loss_elements.shape[-1] == nc:
                    cw_dev = cw_dev.view(1, 1, -1)
                elif loss_elements.shape[1] == nc:
                    cw_dev = cw_dev.view(1, -1, 1)
            elif loss_elements.ndim == 2:
                if loss_elements.shape[-1] == nc:
                    cw_dev = cw_dev.view(1, -1)
                elif loss_elements.shape[0] == nc:
                    cw_dev = cw_dev.view(-1, 1)
        loss_elements = loss_elements * cw_dev

    return loss_elements


class ConfidenceWeightedVFLLoss:
    """Lớp bọc gắn vào hàm mất mát huấn luyện Ultralytics YOLO26 để áp dụng Varifocal Loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

    def create_criterion_callback(self) -> Any:
        """Tạo callback cho Ultralytics Trainer để thay thế hàm mất mát phân loại mặc định bằng VFL."""
        cw = self.class_weights
        alpha = self.alpha
        gamma = self.gamma

        def _callback(trainer: Any) -> None:
            from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

            class CustomVFLDetectionLoss(v8DetectionLoss):
                def __init__(self, model: Any, tal_topk: int = 10, tal_topk2: int | None = None):
                    super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                    self.cw = getattr(model, "class_weights", None) or cw
                    self.alpha = alpha
                    self.gamma = gamma

                def get_assigned_targets_and_loss(self, preds: Any, batch: Any) -> tuple:
                    loss = torch.zeros(3, device=self.device)
                    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

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
                    # Tính toán mất mát phân lớp bằng Varifocal Loss
                    cls_loss_elements = make_vfl_loss_elements(
                        pred_scores=pred_scores,
                        target_scores=target_scores,
                        alpha=self.alpha,
                        gamma=self.gamma,
                        class_weights=self.cw,
                    )
                    loss[1] = cls_loss_elements.sum() / target_scores_sum

                    # Tính toán mất mát bounding box (CIoU + DFL)
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

            det_model = trainer.model
            if cw is not None:
                det_model.class_weights = cw.to(next(det_model.parameters()).device)
            existing_criterion = det_model.init_criterion()
            if isinstance(existing_criterion, E2ELoss):
                det_model.criterion = E2ELoss(det_model, loss_fn=CustomVFLDetectionLoss)
            else:
                det_model.criterion = CustomVFLDetectionLoss(det_model)

        return _callback

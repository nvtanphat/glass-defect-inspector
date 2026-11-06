"""Hàm mất mát phụ trợ Khoảng cách Wasserstein Chuẩn hóa Thích ứng Quy mô (Scale-Aware NWD Loss).

Cài đặt khoảng cách Gaussian Wasserstein-2 với trọng số thích ứng theo kích thước vật thể,
được thiết kế chuyên biệt cho bài toán phát hiện khuyết tật kích thước cực nhỏ trên chai lọ và bề mặt thủy tinh.

Công thức toán học:
    Mỗi bounding box (cx, cy, w, h) được mô hình hóa thành một phân phối Gauss 2 chiều N(mu, Sigma),
    trong đó:
        mu = [cx, cy]^T
        Sigma = diag(w^2 / 4, h^2 / 4)
    Khoảng cách Wasserstein-2 bình phương giữa hộp dự đoán và hộp nhãn thực:
        W_2^2 = ||mu_p - mu_t||^2 + ||Sigma_p^{1/2} - Sigma_t^{1/2}||_F^2
              = (cx_p - cx_t)^2 + (cy_p - cy_t)^2 + 1/4 * ((w_p - w_t)^2 + (h_p - h_t)^2)
    Độ tương đồng chuẩn hóa:
        NWD = exp(-W_2 / C)
    Hàm mất mát:
        Loss_NWD = (1 - NWD) * Weight_Scale
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class ScaleAwareNWDLoss(nn.Module):
    """Hàm mất mát bổ trợ Scale-Aware Normalized Wasserstein Distance (NWD).

    Tham số:
        c (float): Hằng số chuẩn hóa khoảng cách theo pixel (mặc định: 16.0).
        scale_aware (bool): Bật/tắt cơ chế gán trọng số thích ứng quy mô (mặc định: True).
        tau (float): Ngưỡng quy mô khuyết tật đặc trưng theo pixel (mặc định: 32.0).
        alpha_scale (float): Hệ số khuếch đại trọng số cho khuyết tật nhỏ (mặc định: 1.0).
        min_weight (float): Giới hạn chặn dưới trọng số (mặc định: 0.5).
        max_weight (float): Giới hạn chặn trên trọng số (mặc định: 2.0).
        img_size (tuple | list): Kích thước ảnh (height, width) để đổi về pixel (mặc định: 960x960).
        eps (float): Epsilon chống chia cho 0 và đảm bảo ổn định số học (mặc định: 1e-7).
    """

    def __init__(
        self,
        c: float = 16.0,
        scale_aware: bool = True,
        tau: float = 32.0,
        alpha_scale: float = 1.0,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
        img_size: tuple[int, int] | list[int] = (960, 960),
        eps: float = 1e-7,
    ):
        super().__init__()
        self.c = float(c)
        self.scale_aware = bool(scale_aware)
        self.tau = float(tau)
        self.alpha_scale = float(alpha_scale)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.img_size = tuple(img_size)
        self.eps = float(eps)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        num_boxes: float | int | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Tính toán tổn thất NWD thích ứng quy mô giữa các hộp dự đoán và nhãn thực.

        Tham số:
            pred_boxes: Tensor (N, 4) tọa độ tâm và kích thước chuẩn hóa [0, 1] dạng (cx, cy, w, h).
            target_boxes: Tensor (N, 4) tọa độ nhãn thực chuẩn hóa [0, 1] dạng (cx, cy, w, h).
            num_boxes: Số lượng hộp mục tiêu trong toàn batch để tính trung bình.

        Trả về:
            Tensor vô hướng biểu diễn giá trị tổn thất Scale-Aware NWD.
        """
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        # Bước 1: Quy đổi tọa độ chuẩn hóa [0, 1] về tọa độ điểm ảnh pixel thực tế (cx, cy, w, h)
        h_img, w_img = self.img_size[0], self.img_size[1]
        scale_factor = pred_boxes.new_tensor([w_img, h_img, w_img, h_img])

        pred_px = pred_boxes * scale_factor
        tgt_px = target_boxes * scale_factor

        cx_p, cy_p, w_p, h_p = pred_px.unbind(-1)
        cx_t, cy_t, w_t, h_t = tgt_px.unbind(-1)

        # Bước 2: Tính khoảng cách Wasserstein-2 bình phương giữa hai phân phối Gauss 2 chiều
        # Khoảng cách tâm: (cx_p - cx_t)^2 + (cy_p - cy_t)^2
        center_dist_sq = (cx_p - cx_t).pow(2) + (cy_p - cy_t).pow(2)
        # Khoảng cách kích thước Frobenius ma trận hiệp phương sai: 0.25 * ((w_p - w_t)^2 + (h_p - h_t)^2)
        wh_dist_sq = 0.25 * ((w_p - w_t).pow(2) + (h_p - h_t).pow(2))
        w2_sq = center_dist_sq + wh_dist_sq

        # Bước 3: Khoảng cách Wasserstein căn bậc hai có bù epsilon
        w_dist = torch.sqrt(torch.clamp(w2_sq, min=0.0) + self.eps)

        # Bước 4: Độ tương đồng NWD theo hàm mũ thuộc khoảng (0, 1]
        nwd = torch.exp(-w_dist / self.c)

        # Tổn thất cơ bản: 1 - NWD (trừ độ lệch zero_offset để khi hai hộp trùng nhau thì loss = 0)
        zero_offset = 1.0 - math.exp(-math.sqrt(self.eps) / self.c)
        loss_nwd = torch.clamp(1.0 - nwd - zero_offset, min=0.0)

        # Bước 5: Trọng số thích ứng quy mô (Scale-Aware Weighting)
        if self.scale_aware:
            # Quy mô khuyết tật: trung bình nhân chiều rộng và chiều cao nhãn thực theo pixel
            target_scale = torch.sqrt(torch.clamp(w_t * h_t, min=self.eps))
            # Trọng số hàm mũ suy giảm: vật thể càng nhỏ (scale << tau) thì trọng số càng cao
            scale_weights = 1.0 + self.alpha_scale * torch.exp(-target_scale / self.tau)
            scale_weights = torch.clamp(scale_weights, min=self.min_weight, max=self.max_weight)
            # Ngắt gradient flow qua trọng số quy mô để ổn định huấn luyện
            loss_nwd = loss_nwd * scale_weights.detach()

        # Bước 6: Chuẩn hóa theo số lượng hộp trong batch
        if isinstance(num_boxes, (int, float)):
            divisor = max(float(num_boxes), 1.0)
        else:
            divisor = torch.clamp(num_boxes, min=1.0)

        return loss_nwd.sum() / divisor

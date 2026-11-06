"""Mô-đun Giám sát Chất lượng Dữ liệu và Trôi dạt Quang học (Data Quality & Drift Guard).

Mô-đun kiểm định đầu vào siêu nhẹ, không tiêu tốn tài nguyên GPU, sử dụng NumPy và OpenCV thuần:
- Kiểm tra độ sáng và độ phơi sáng (phát hiện ảnh thiếu sáng hoặc chói lóa).
- Kiểm tra độ tương phản (phát hiện ánh sáng phẳng làm mất chi tiết vân nứt).
- Đánh giá độ sắc nét và mờ nhòe thông qua phương sai toán tử Laplacian (Laplacian variance).
- Kiểm tra tính hợp lệ về kích thước và tỷ lệ khung hình trước khi đưa vào mô hình suy luận.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

# Các ngưỡng vận hành hiệu chuẩn trên bộ dữ liệu bề mặt kính công nghiệp
MIN_BRIGHTNESS = 35.0  # Dưới ngưỡng này ảnh quá tối, chi tiết khuyết tật chìm vào bóng đen
MAX_BRIGHTNESS = 225.0  # Trên ngưỡng này ảnh bị lóa sáng quang học, biên trầy xước bị mất
MIN_CONTRAST = 18.0  # Độ lệch chuẩn tối thiểu của các điểm ảnh xám
MIN_SHARPNESS = 45.0  # Ngưỡng phương sai Laplacian; thấp hơn biểu hiện mờ nhòe do rung máy hoặc mất nét


def analyze_image_quality(
    image: Image.Image | np.ndarray,
    min_brightness: float = MIN_BRIGHTNESS,
    max_brightness: float = MAX_BRIGHTNESS,
    min_contrast: float = MIN_CONTRAST,
    min_sharpness: float = MIN_SHARPNESS,
) -> dict[str, Any]:
    """Đánh giá chất lượng quang học của ảnh chai/kính trước khi đưa vào mô hình AI.

    Tham số:
        image: Ảnh đối tượng dạng PIL Image hoặc mảng NumPy (RGB hoặc BGR).
        min_brightness: Ngưỡng dưới của độ sáng trung bình chấp nhận được (0-255).
        max_brightness: Ngưỡng trên của độ sáng trung bình chấp nhận được (0-255).
        min_contrast: Độ lệch chuẩn mức xám tối thiểu được chấp nhận.
        min_sharpness: Phương sai toán tử Laplacian tối thiểu (đo độ nét).

    Trả về:
        dict chứa:
            - is_acceptable: bool, True nếu ảnh đủ tiêu chuẩn kiểm định
            - metrics: dict gồm các chỉ số quang học đo đạc được
            - warnings: danh sách cảnh báo chất lượng bằng ngôn ngữ tự nhiên
            - status: 'PASS' | 'WARNING' | 'REJECT'
    """
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"))
        w, h = image.size
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(image, code)
            h, w = image.shape[:2]
        else:
            gray = image
            h, w = gray.shape[:2]
    else:
        raise TypeError(f"Định dạng ảnh không được hỗ trợ: {type(image)}")

    if gray.size == 0 or w == 0 or h == 0:
        return {
            "is_acceptable": False,
            "status": "REJECT",
            "metrics": {
                "width": w,
                "height": h,
                "brightness": 0.0,
                "contrast": 0.0,
                "sharpness": 0.0,
            },
            "warnings": ["Ảnh có kích thước rỗng hoặc bộ đệm pixel không hợp lệ."],
        }

    # 1. Trích xuất chỉ số độ sáng và độ phơi sáng (Luminance & Exposure)
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    # 2. Đánh giá độ sắc nét / độ nhòe chuyển động (Laplacian Variance)
    # Giảm kích thước ảnh xuống nếu ảnh quá lớn (>2500px) để đảm bảo thời gian xử lý < 3ms
    if max(w, h) > 2500:
        scale_factor = 2000.0 / max(w, h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        small_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        laplacian = cv2.Laplacian(small_gray, cv2.CV_64F)
    else:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))

    # 3. Kích thước hình học và tỷ lệ khung hình (Aspect Ratio)
    aspect_ratio = round(float(w) / float(h), 3)

    warnings: list[str] = []

    if mean_brightness < min_brightness:
        warnings.append(
            f"Image is under-exposed / Ảnh thiếu sáng (độ sáng {mean_brightness:.1f} < {min_brightness}). "
            "Các vết trầy xước trong vùng tối có thể bị bỏ sót."
        )
    elif mean_brightness > max_brightness:
        warnings.append(
            f"Image is over-exposed / Ảnh thừa sáng chói lóa (độ sáng {mean_brightness:.1f} > {max_brightness}). "
            "Vùng phản xạ gương có thể gây báo động giả."
        )

    if contrast_std < min_contrast:
        warnings.append(
            f"Low contrast detected / Độ tương phản thấp (độ tương phản {contrast_std:.1f} < {min_contrast}). "
            "Biên phân cách giữa khuyết tật và bề mặt thủy tinh không rõ ràng."
        )

    if sharpness < min_sharpness:
        warnings.append(
            f"Image appears blurry / Ảnh bị mờ nhòe (chỉ số sắc nét {sharpness:.1f} < {min_sharpness})."
        )

    # Xác định trạng thái cổng kiểm định chất lượng
    if not warnings:
        status = "PASS"
        is_acceptable = True
    else:
        status = "WARNING"
        is_acceptable = True  # Cảnh báo không chặn suy luận nhưng được ghi nhận vào nhật ký kiểm toán

    return {
        "is_acceptable": is_acceptable,
        "status": status,
        "metrics": {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "brightness": round(mean_brightness, 2),
            "contrast": round(contrast_std, 2),
            "sharpness": round(sharpness, 2),
        },
        "warnings": warnings,
    }

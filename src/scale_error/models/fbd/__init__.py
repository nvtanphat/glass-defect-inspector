"""Các mô-đun kiến trúc lấy cảm hứng từ FBDNet tích hợp cho D-FINE-M (FBDNet-M).

Bao gồm:
- AFTA: Chú ý miền tần số / mục tiêu thích ứng tại backbone stages.
- SFIF: Hợp nhất tương tác miền không gian - miền tần số bằng 2D FFT.
- DCFSB: Khối lọc đặc trưng đa tỉ lệ dày đặc bảo vệ tín hiệu vi mô.
"""

from .afta import AFTA
from .dcfsb import DCFSB
from .sfif import SFIF

__all__ = ["AFTA", "SFIF", "DCFSB"]

// Display names only; persisted model identities and quality decisions stay unchanged.
const QUALITY_NAMES = {PASS: 'Đạt', REJECT: 'Không đạt', REVIEW: 'Cần xác nhận'};
const STATUS_NAMES = {PENDING: 'Đang chờ', PROCESSING: 'Đang kiểm tra', COMPLETED: 'Đã hoàn thành', FAILED: 'Không hoàn thành'};
const DEFECT_NAMES = {DiVat: 'Dị vật', DiVatLoiLom: 'Dị vật lồi lõm', LoiChi: 'Nứt chỉ', LoiNhua: 'Ba via nhựa', LoiTray: 'Trầy xước'};
const MODEL_NAMES = {
  'apex-triad-ensemble': '👑 Adaptive Triad Ensemble SOTA (46.10% mAP)',
  'yolo26m-distilled': 'Distilled Student YOLO26m (~48 FPS, 41.01% mAP)',
  'fbdnet-m': 'FBDNet-M SOTA Đơn Lẻ (42.42% mAP)',
  'dfine-m': 'D-FINE-M (41.34% mAP)',
  'yolo26m-roi-vfl': 'YOLO26m + ROI + VFL (40.09% mAP)',
  'yolo26m-base': 'YOLO26m Base',
  'yolo11m': 'YOLO11m Baseline',
  'rtdetr-l': 'RT-DETR-L Baseline',
};
function qualityLabel(value) { return QUALITY_NAMES[value] || value; }
function statusLabel(value) { return STATUS_NAMES[value] || value; }
function defectLabel(value) { return DEFECT_NAMES[value] || value; }
function modelLabel(value) { return MODEL_NAMES[value] || value || 'Bộ kiểm tra'; }
function durationLabel(ms) { return ms == null ? 'Chưa có kết quả' : `${(ms / 1000).toLocaleString('vi-VN', {maximumFractionDigits: 2})} giây`; }
function assessmentText(detail) {
  if (detail.reviews.length) return detail.explanation;
  if (detail.status === 'FAILED') return 'Lượt kiểm tra chưa hoàn thành. Vui lòng kiểm tra lại ảnh và thử lại.';
  if (detail.status !== 'COMPLETED') return 'Ảnh đang được xử lý. Vui lòng chờ.';
  return detail.conclusion === 'REJECT'
    ? 'Đã phát hiện dấu hiệu lỗi trên ảnh. Hãy xem các vị trí được đánh dấu trước khi xác nhận.'
    : 'Chưa phát hiện lỗi ở mức kiểm tra đã chọn. Cần người kiểm tra xác nhận trước khi kết luận đạt.';
}

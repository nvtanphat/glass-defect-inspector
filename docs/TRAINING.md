# Training và evaluation local

Chạy từ root sau khi cài `python -m pip install -e ".[app,dev]"`. `scripts/train.py` gọi CLI/trainer hiện có, không khởi động web hoặc database.

## Dữ liệu

Dataset YOLO gồm `train`, `valid`, `test`, mỗi split có `images/` và `labels/`. YAML runtime khai báo đường dẫn split và mapping: `DiVat`, `DiVatLoiLom`, `LoiChi`, `LoiNhua`, `LoiTray`.

Nếu tải bằng Roboflow, cung cấp `ROBOFLOW_API_KEY` qua môi trường:

```bash
python scripts/train.py prepare --output data/raw --split-output data/processed
```

Lệnh dùng dataset v26, chia lại validation gốc theo implementation hiện có. Kiểm tra báo cáo nhãn/split trước khi train. Test không dùng để chỉnh model. ROI dùng `processing/roi_config.json`; ghi rõ có crop hay không khi đánh giá.

## YOLO và RT-DETR

```bash
python scripts/train.py train --data data/processed/runtime_data.yaml --config configs/train_yolo26m.yaml --project-dir artifacts/runs
python scripts/train.py evaluate --data data/processed/runtime_data.yaml --weights artifacts/runs/yolo26m/weights/best.pt --output artifacts/evaluation/yolo26m.json
python scripts/train.py predict --weights artifacts/runs/yolo26m/weights/best.pt --source path/to/images --output artifacts/predictions
```

Có thể thay config bằng `configs/train_yolo11m.yaml` hoặc `configs/train_rtdetr.yaml`. Chọn đường dẫn weights thực tế từ output; chỉnh device, batch size, epochs và image size trong YAML theo máy local.

## D-FINE-M

Cung cấp runtime tại `results/D-FINE/` và pretrained weights. Wrapper chuyển nhãn sang COCO, tạo config và gọi framework training riêng:

```bash
python scripts/train.py train-dfine --data data/processed/runtime_data.yaml --dfine-repo results/D-FINE --weights weights/dfine_m_coco.pth --output-dir artifacts/runs/dfine_m --epochs 100 --batch-size 4
python scripts/train.py evaluate-dfine --data data/processed/runtime_data.yaml --dfine-repo results/D-FINE --weights artifacts/runs/dfine_m/best_stg2.pth --output artifacts/evaluation/dfine_m.json
```

Chọn checkpoint tồn tại thực tế: `best.pth`, `best_stg1.pth` hoặc `best_stg2.pth`. Resume dùng `train-dfine --resume path/to/last.pth`, cần optimizer/scheduler và runtime tương thích.

## FBDNet E0/E1

Runtime cần HGNetv2FBD, FBDHybridEncoder và loss NWD tương ứng; clone D-FINE upstream đơn thuần chưa đủ. Hai config thực thi phải tồn tại ở `results/D-FINE/configs/dfine/custom/fbdnet_m_e0_baseline_20ep.yml` và `fbdnet_m_e1_sanwd_20ep.yml`, với đường dẫn COCO/images local hợp lệ. YAML trong `configs/experiments/` mô tả thí nghiệm; runner dùng trực tiếp config D-FINE của runtime.

```bash
python scripts/train_fbdnet.py --exp both --weights weights/dfine_m_fbdnet_best.pth --epochs 20 --batch-size 8 --device cuda --output-dir runs/fbdnet_nwd
```

`--exp e0` chạy baseline; `--exp e1` thêm Scale-Aware NWD. Đây là fine-tuning từ weights với optimizer/scheduler mới, không phải train từ đầu hoặc resume. Nhiều GPU CUDA dùng distributed launcher. Thiếu checkpoint/config/runtime thì dừng. `--help` không train.

## AHRD

Runtime cần config/module AHRD tương ứng trong `results/D-FINE/`:

```bash
python scripts/run_ahrd_experiment.py --exp e1 --weights weights/dfine_m_coco.pth --dry-run
python scripts/run_ahrd_experiment.py --exp both --weights weights/dfine_m_coco.pth --seeds 42 43 44
```

Dry-run chỉ kiểm tra đường dẫn/config; không chứng minh model đã load. Full runner train nhiều seed liên tiếp, chỉ chạy khi đã chuẩn bị tài nguyên.

## Evaluation FBDNet đầy đủ

```bash
python scripts/evaluate_fbdnet_full.py --device cuda --batch-size 4 --output artifacts/evaluation/fbdnet_m
```

Evaluator dùng checkpoint/ROI local, kiểm tra protocol test 890 ảnh và 1.573 boxes sau ROI, lưu COCO metrics, confusion, curves, Grad-CAM và provenance. `--limit` chỉ là smoke subset, không báo là full-test result. Giữ output từng lần riêng. Benchmark trong tài liệu là số liệu lịch sử, không cam kết tái lập trên mọi runtime.

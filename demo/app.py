from __future__ import annotations

import os

import gradio as gr
from ultralytics import YOLO

WEIGHTS = os.environ.get("YOLO_WEIGHTS", "artifacts/runs/yolo26m_960/weights/best.pt")
NMS = os.environ.get("YOLO_NMS", "true").lower() not in ("0", "false", "no")
model = YOLO(WEIGHTS)


def infer(image):
    result = model.predict(image, imgsz=960, conf=0.25, nms=NMS, verbose=False)[0]
    return result.plot()


demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="numpy", label="Input image"),
    outputs=gr.Image(type="numpy", label="YOLO26 prediction"),
    title="Scale_Error_DC — YOLO26m",
)

if __name__ == "__main__":
    demo.launch()

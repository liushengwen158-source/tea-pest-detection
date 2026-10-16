"""把训练好的权重导出为部署格式。

用法：python scripts/export.py [格式]
默认导出 onnx。确定部署硬件后再换格式重跑即可，不需要重新训练。

格式与对应硬件：
    onnx      x86 CPU / 通用，OpenVINO 前的中间格式
    openvino  Intel CPU / 核显
    engine    NVIDIA Jetson（TensorRT，需在该设备上导出）
    ncnn      树莓派、安卓
    rknn      瑞芯微 RK3588
    tflite    安卓 / 边缘 TPU
"""

import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "detect" / "tea_yolov8n" / "weights" / "best.pt"

FORMATS = ("onnx", "torchscript", "openvino", "engine", "ncnn", "rknn", "tflite")


def main() -> None:
    fmt = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    if fmt not in FORMATS:
        print(f"支持的格式：{', '.join(FORMATS)}")
        return

    if not WEIGHTS.exists():
        print(f"找不到训练好的权重：{WEIGHTS}")
        print("请先运行：python scripts/train.py")
        return

    model = YOLO(str(WEIGHTS))
    path = model.export(format=fmt, imgsz=640)
    print(f"导出完成：{path}")


if __name__ == "__main__":
    main()


"""害虫检测模型训练脚本（YOLOv8n）。

数据集：data/tea —— 8 类茶叶病虫害。
用法：python scripts/train.py
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "tea" / "data.yaml"

CONFIG = {
    "model": str(ROOT / "weights" / "yolov8n.pt"),  # COCO 预训练权重，做迁移学习
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,  # 显存不够就改 8 或 -1（自动批大小）
    "device": 0,  # 0 = RTX 4050；没有可用 GPU 时改成 "cpu"
    "workers": 8,  # Windows 下 0 会拖慢数据加载；若崩溃再降回 4 或 0
    "cache": "ram",  # 数据集很小，全部缓存进内存加速
    "patience": 30,  # 30 轮无提升就早停
    "close_mosaic": 10,  # 最后 10 轮关闭 mosaic 增强，提升收敛
    "project": str(ROOT / "runs" / "detect"),
    "name": "tea_yolov8n",
}


def main() -> None:
    model = YOLO(CONFIG.pop("model"))
    model.train(data=str(DATA), **CONFIG)

    # 训练结束后再跑一次验证，拿到明确的最终指标
    metrics = model.val(data=str(DATA))
    print(f"\nmAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()



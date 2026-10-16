"""害虫检测推理：图片 / 视频 / 摄像头 / 文件夹。

用法：
    python scripts/predict.py 图片.jpg
    python scripts/predict.py 视频.mp4
    python scripts/predict.py 0            # 打开摄像头实时检测
    python scripts/predict.py 文件夹路径    # 批量处理
    python scripts/predict.py              # 不传参数则处理 test_images/（里面没图才退回测试集一张）

结果默认保存到 runs/detect/predict/，输出图片会被重命名为 1.jpg、2.jpg……
方便按顺序查看；每次运行会先清空该目录，保证序号从 1 开始。
"""

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "detect" / "tea_yolov8n" / "weights" / "best.pt"
OUT_DIR = ROOT / "runs" / "detect" / "predict"

CONF = 0.25  # 置信度阈值，漏检多就调低，误检多就调高
IOU = 0.45  # NMS 的 IoU 阈值
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rename_sequential(directory: Path) -> None:
    """把目录里的图片按顺序重命名为 1、2、3 ……（视频等非图片不动）。"""
    imgs = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)
    tmp = []
    for p in imgs:  # 先改成临时名，避免和已有的 1.jpg 冲突
        t = p.with_name("__tmp__" + p.name)
        p.rename(t)
        tmp.append(t)
    for i, t in enumerate(tmp, start=1):
        t.rename(t.with_name(f"{i}{t.suffix.lower()}"))


def main() -> None:
    if len(sys.argv) < 2:
        # 不传参数时，优先处理根目录 test_images/ 里的图片；没有才退回测试集一张
        user_dir = ROOT / "test_images"
        user_imgs = (
            sorted(p for p in user_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            if user_dir.exists()
            else []
        )
        if user_imgs:
            source = str(user_dir)  # 整个文件夹，全部处理
            print(f"未指定输入，默认处理 {user_dir}（共 {len(user_imgs)} 张）\n")
        else:
            ds_dir = ROOT / "data" / "tea" / "test" / "images"
            ds_imgs = sorted(ds_dir.glob("*.jpg"))
            if not ds_imgs:
                print("未找到默认图片，请传入路径：python scripts/predict.py 图片.jpg")
                return
            source = str(ds_imgs[0])
            print(f"未指定输入，默认使用测试集图片：{source}\n")
    else:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg

    if not WEIGHTS.exists():
        print(f"找不到训练好的权重：{WEIGHTS}")
        print("请先运行：python scripts/train.py")
        return

    # 清空上一次的结果，保证序号从 1 开始、不残留旧图
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    model = YOLO(str(WEIGHTS))
    model.predict(
        source=source,
        conf=CONF,
        iou=IOU,
        save=True,  # 保存带框的结果
        show=isinstance(source, int),  # 摄像头模式自动弹窗实时显示
        project=str(ROOT / "runs" / "detect"),
        name="predict",
    )

    renamed = (
        [p for p in OUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
        if OUT_DIR.exists()
        else []
    )
    if not isinstance(source, int) and renamed:
        rename_sequential(OUT_DIR)
        print(f"\n结果已保存到 {OUT_DIR}")
        print(f"共 {len(renamed)} 张，已重命名为 1、2、3 ……")


if __name__ == "__main__":
    main()


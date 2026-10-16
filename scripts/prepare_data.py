"""把 HuggingFace parquet 数据集转换成 YOLO 目录格式。

来源：Project-AgML/tea_leaf_disease_pest_detection（= figshare《Tea-Leaf Disease
and Pest Detection Dataset v1.0》的 HF 镜像），8 类、9,591 张，CC BY 4.0。

输出：
    data/tea/{train,valid,test}/images/*.jpg
    data/tea/{train,valid,test}/labels/*.txt   （YOLO：cls cx cy w h，归一化）
    data/tea/data.yaml

用法：
    python scripts/prepare_data.py           # 先诊断格式，再转换
    python scripts/prepare_data.py --dry      # 只诊断，不写文件
"""

import io
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "_raw.parquet"
OUT = ROOT / "data" / "tea"

NAMES = [
    "Black rot of tea",
    "Brown blight of tea",
    "Leaf rust of tea",
    "Red spider infested tea leaf",
    "Tea mosquito bug infested leaf",
    "Tea leaf",
    "White spot of tea",
    "disease",
]

SPLIT_MAP = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _image_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"未知 image 字段类型：{type(cell)}")


def _objects(cell):
    if isinstance(cell, dict):
        return list(cell.get("bbox", [])), list(cell.get("categories", []))
    raise TypeError(f"未知 objects 字段类型：{type(cell)}")


def main():
    dry = "--dry" in sys.argv
    df = pd.read_parquet(RAW)
    print(f"总行数：{len(df)}")
    print(f"列：{list(df.columns)}")

    r0 = df.iloc[0]
    boxes, cats = _objects(r0["objects"])
    im = Image.open(io.BytesIO(_image_bytes(r0["image"])))
    W, H = im.size
    print(f"\n[诊断] 首行 split={r0['split']!r}  图像尺寸={W}x{H}")
    print(f"[诊断] 首行框数={len(boxes)}  前 3 个框={boxes[:3]}")
    print(f"[诊断] 首行类别 id={cats[:3]}")

    all_b = [c for _, row in df.head(200).iterrows() for c in _objects(row["objects"])[0]]
    mx = max((max(b) for b in all_b), default=0)
    mn = min((min(b) for b in all_b), default=0)
    print(f"[诊断] 前 200 行框坐标范围：min={mn:.3f} max={mx:.3f}")
    absolute = mx > 1.5
    print(f"[诊断] 判定框格式：{'绝对像素 [x,y,w,h]' if absolute else '已归一化'}")
    print("\n[诊断] 原始 split 分布：", dict(Counter(df["split"].astype(str))))

    if dry:
        print("\n--dry：仅诊断，未写文件。")
        return

    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    for sp in ("train", "valid", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    counters = Counter()
    for i, (_, row) in enumerate(df.iterrows()):
        sp = SPLIT_MAP.get(str(row["split"]).strip().lower())
        if sp is None:
            counters["跳过"] += 1
            continue
        im = Image.open(io.BytesIO(_image_bytes(row["image"]))).convert("RGB")
        W, H = im.size
        name = f"{sp}_{i:06d}"
        im.save(OUT / sp / "images" / f"{name}.jpg", quality=95)
        boxes, cats = _objects(row["objects"])
        lines = []
        for b, c in zip(boxes, cats):
            x, y, w, h = b
            if absolute:
                x, y, w, h = x / W, y / H, w / W, h / H
            cx, cy = x + w / 2, y + h / 2
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            w, h = min(max(w, 0), 1), min(max(h, 0), 1)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counters["框"] += 1
        (OUT / sp / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        counters[sp] += 1

    yaml_txt = "path: '{}'\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: {}\nnames:\n{}".format(
        OUT.as_posix(), len(NAMES), "\n".join(f"- {n}" for n in NAMES)
    )
    (OUT / "data.yaml").write_text(yaml_txt + "\n", encoding="utf-8")

    print("\n转换完成：")
    for sp in ("train", "valid", "test"):
        print(f"  {sp:5s}: {counters[sp]} 张")
    print(f"  合计框数：{counters['框']}")
    if counters["跳过"]:
        print(f"  跳过未知 split：{counters['跳过']}")
    print(f"\ndata.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()


import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import shutil

# 设置数据集缓存路径，避免占用C盘
fo.config.dataset_zoo_dir = r"E:\datasets"

# 下载指定类别的图片
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Man", "Woman"],
    max_samples=100,
    shuffle=True,
    seed=42
)

# 创建原始图片文件夹
save_dir = Path("dataset_people/raw_images")
save_dir.mkdir(parents=True, exist_ok=True)

# 将图片复制到自己的项目中
for sample in dataset:
    src = Path(sample.filepath)
    dst = save_dir / src.name
    shutil.copy2(src, dst)

print(f"下载完成，共 {len(dataset)} 张图片")

import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# 1. 定义路径
root = Path("dataset_people")

raw_dir = root / "raw_images"
image_dir = root / "images"
label_dir = root / "labels"

csv_path = Path(
    r"E:\datasets\open-images-v7\validation\labels\detections.csv"
)

# 2. 创建输出文件夹
image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# 3. 类别映射，与手工数据集保持一致
class_map = {
    "/m/04yx4": 0,      # man
    "/m/03bt1vf": 1     # woman
}

# 4. 获取已经下载的图片
images = {
    p.stem: p
    for p in raw_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
}

print("原始图片数量:", len(images))

# 存储每张图片的所有标签
annotations = defaultdict(list)

# 不适合训练的图片
excluded = set()

# 5. 读取 Google CSV
with open(csv_path, "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        image_id = row["ImageID"]
        label_name = row["LabelName"]

        # 只处理已经下载的图片
        if image_id not in images:
            continue

        # 只处理需要的两个类别
        if label_name not in class_map:
            continue

        # 排除群体标注、绘画等样本
        if (
            row["IsGroupOf"] == "1"
            or row["IsDepiction"] == "1"
        ):
            excluded.add(image_id)
            continue

        class_id = class_map[label_name]

        # 6. 读取归一化边界框坐标
        xmin = float(row["XMin"])
        xmax = float(row["XMax"])
        ymin = float(row["YMin"])
        ymax = float(row["YMax"])

        # 检查坐标是否合法
        if not (
            0 <= xmin < xmax <= 1
            and 0 <= ymin < ymax <= 1
        ):
            excluded.add(image_id)
            continue

        # 转换为 YOLO 格式
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        line = (
            f"{class_id} {xc:.6f} {yc:.6f} "
            f"{w:.6f} {h:.6f}"
        )

        annotations[image_id].append(line)

# 7. 复制图片，生成 TXT 标签
count = 0
classes = Counter()

for image_id, lines in annotations.items():

    if image_id in excluded:
        continue

    image_path = images[image_id]

    shutil.copy2(
        image_path,
        image_dir / image_path.name
    )

    txt_path = label_dir / (image_id + ".txt")

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    for line in lines:
        classes[int(line.split()[0])] += 1

    count += 1

# 8. 输出统计结果
print("\n转换完成！")
print("有效图片数量:", count)
print("跳过图片数量:", len(images) - count)
print("目标总数:", sum(classes.values()))
print("man 数量:", classes[0])
print("woman 数量:", classes[1])

import json
import shutil
from pathlib import Path

# 数据集路径
root = Path("dataset_private")

raw_dir = root / "raw_images"
image_dir = root / "images"
label_dir = root / "labels"

# 创建输出文件夹
image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# 类别映射
class_map = {
    "man": 0,
    "woman": 1
}


def convert(json_path):
    # 1. 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W = data["imageWidth"]
    H = data["imageHeight"]

    lines = []

    # 2. 遍历图片中的所有标注目标
    for shape in data["shapes"]:

        if shape["shape_type"] != "rectangle":
            raise ValueError(f"非矩形标注: {json_path}")

        class_id = class_map[shape["label"]]

        (x1, y1), (x2, y2) = shape["points"]

        # 3. 坐标转换及归一化
        xc = (x1 + x2) / (2 * W)
        yc = (y1 + y2) / (2 * H)

        w = abs(x2 - x1) / W
        h = abs(y2 - y1) / H

        # 检查坐标是否合法
        if not (
            0 <= min(x1, x2) < max(x1, x2) <= W
            and 0 <= min(y1, y2) < max(y1, y2) <= H
        ):
            raise ValueError(f"非法标注框: {json_path}")

        # 4. 生成 YOLO 格式的一行标签
        line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        lines.append(line)

    # 5. 找到对应的原始图片
    image_path = json_path.parent / data["imagePath"]

    # 6. 复制图片
    shutil.copy2(image_path, image_dir / image_path.name)

    # 7. 写入 TXT 标签文件
    txt_path = label_dir / (image_path.stem + ".txt")

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"{image_path.name}: {len(lines)} objects")

    return len(lines)


# 遍历所有已经标注的 JSON 文件
json_files = sorted(raw_dir.glob("*.json"))

total_objects = 0

for json_path in json_files:
    total_objects += convert(json_path)

print("\n转换完成！")
print("图片数量:", len(json_files))
print("目标数量:", total_objects)

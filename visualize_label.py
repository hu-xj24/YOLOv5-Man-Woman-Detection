
import cv2
from pathlib import Path

# 数据集路径
roots = [
    Path("dataset_private"),
    Path("dataset_people")
]

# 类别映射
class_names = {
    0: "man",
    1: "woman"
}

# OpenCV 使用 BGR 颜色格式
colors = {
    0: (255, 0, 0),      # 蓝色
    1: (0, 0, 255)       # 红色
}


def draw_labels(image_path, label_path):

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    H, W = img.shape[:2]

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:

        if not line.strip():
            continue

        cls, xc, yc, w, h = map(float, line.split())
        cls = int(cls)

        # 反归一化
        xmin = int((xc - w / 2) * W)
        xmax = int((xc + w / 2) * W)
        ymin = int((yc - h / 2) * H)
        ymax = int((yc + h / 2) * H)

        color = colors[cls]
        name = class_names[cls]

        # 绘制矩形框
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2
        )

        # 绘制类别名称
        cv2.putText(
            img,
            name,
            (xmin, max(20, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return img


# 收集两个数据集中的图片和标签
samples = []

for root in roots:

    image_dir = root / "images"
    label_dir = root / "labels"

    for image_path in sorted(image_dir.glob("*")):

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            print("缺少标签:", image_path)
            continue

        samples.append((image_path, label_path))

print("图片总数:", len(samples))

# 交互式查看图片
index = 0

while 0 <= index < len(samples):

    image_path, label_path = samples[index]

    img = draw_labels(image_path, label_path)

    # 限制显示尺寸，避免原图太大
    H, W = img.shape[:2]
    scale = min(1200 / W, 800 / H, 1.0)

    img = cv2.resize(
        img,
        (max(1, int(W * scale)), max(1, int(H * scale)))
    )

    print(f"[{index + 1}/{len(samples)}] {image_path}")

    cv2.imshow("YOLO Label Visualization", img)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("n"):
        index += 1

    elif key == ord("p"):
        index = max(0, index - 1)

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()

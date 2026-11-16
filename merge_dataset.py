import random
import shutil
from pathlib import Path
from collections import Counter

private_root = Path("dataset_private")
google_root = Path("dataset_people")
output_root = Path("dataset_merged")

random.seed(42)


# 1. 收集图片与标签
def collect_samples(root, prefix):

    samples = []

    for img in sorted((root / "images").iterdir()):

        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label = root / "labels" / (img.stem + ".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        samples.append((img, label, prefix))

    return samples


private = collect_samples(private_root, "private")
google = collect_samples(google_root, "google")

print("Private images:", len(private))
print("Google images:", len(google))


# 2. 随机划分Google数据集
random.shuffle(google)

n = len(google)

n_train = int(n * 0.8)
n_val = int(n * 0.1)

datasets = {
    "train": private + google[:n_train],
    "val": google[n_train:n_train + n_val],
    "test": google[n_train + n_val:]
}


# 3. 创建输出目录，防止混入旧数据
output_root.mkdir(exist_ok=False)

for split, samples in datasets.items():

    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    classes = Counter()

    for img, label, prefix in samples:

        # 添加前缀，避免两个数据集文件名冲突
        name = f"{prefix}_{img.stem}"

        shutil.copy2(
            img,
            image_dir / (name + img.suffix)
        )

        shutil.copy2(
            label,
            label_dir / (name + ".txt")
        )

        # 统计目标类别数量
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                classes[int(line.split()[0])] += 1

    print(f"\n{split}:")
    print("images:", len(samples))
    print("man:", classes[0])
    print("woman:", classes[1])

print("\nDataset merge completed!")

import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
FOLDERS = {"train": "seg_train/seg_train", "test": "seg_test/seg_test"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_CALIBRATION = 200
CALIBRATION_SEED = 0
EXPECTED_SPLIT = {"train": (1200, 11993084), "test": (1502, 33351444)}
EXPECTED_CALIBRATION = {"count": 200, "sum": 1921470,
                        "classes": {"buildings": 40, "forest": 35, "glacier": 36,
                                    "mountain": 33, "sea": 29, "street": 27}}

def number(file):
    return int(file.removesuffix(".jpg"))

def read_split(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def part_rows(split, part):
    return [row for row in split if row["part"] == part]

def calibration_rows(split):
    train = sorted(part_rows(split, "train"), key=lambda row: row["file"])
    idx = np.random.default_rng(CALIBRATION_SEED).permutation(len(train))[:N_CALIBRATION]
    return [train[i] for i in idx]

def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_images(data_root, rows, part, transform):
    x = []
    for row in rows:
        with Image.open(data_root / FOLDERS[part] / row["label"] / row["file"]) as image:
            x.append(transform(image.convert("RGB")).numpy())
    y = np.array([CLASSES.index(row["label"]) for row in rows])
    return np.stack(x), y

def split_check(split):
    result = {}
    for part, expected in EXPECTED_SPLIT.items():
        rows = part_rows(split, part)
        got = (len(rows), sum(number(row["file"]) for row in rows))
        result[part] = {"count": got[0], "sum": got[1], "ok": got == expected}
    return result

def calibration_check(rows):
    classes = Counter(row["label"] for row in rows)
    result = {"count": len(rows), "sum": sum(number(row["file"]) for row in rows),
              "classes": {label: classes[label] for label in CLASSES}}
    result["ok"] = result == EXPECTED_CALIBRATION
    return result

def cell(count, total):
    return f"{count} / {total:,}".replace(",", " ")

def report(split):
    checks = {"split": split_check(split), "calibration": calibration_check(calibration_rows(split))}
    print(f"{'Частина':10} {'Кількість / сума імен':>24} {'Збіг':>6}")
    for part, c in checks["split"].items():
        print(f"{part:10} {cell(c['count'], c['sum']):>24} {'так' if c['ok'] else 'ні':>6}")
    c = checks["calibration"]
    print(f"{'calibration':10} {cell(c['count'], c['sum']):>24} {'так' if c['ok'] else 'ні':>6}")
    print("Калібрувальна вибірка за класами: " + ", ".join(f"{k} {v}" for k, v in c["classes"].items()))
    ok = all(v["ok"] for v in checks["split"].values()) and c["ok"]
    print(f"\nПоділ і калібрувальна вибірка збігаються з контрольними числами: {'так' if ok else 'НІ'}")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    args = parser.parse_args()
    report(read_split(args.split))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
PARTS = ["train", "validation", "test"]
FOLDERS = {"train": "seg_train/seg_train", "validation": "seg_test/seg_test", "test": "seg_test/seg_test"}
N_TRAIN_PER_CLASS = 200
SEED = 0
IMG_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_CSV = Path("split.csv")

EXPECTED = {
    "buildings": [(200, 1996761), (218, 4837323), (219, 4837559)],
    "forest": [(200, 2032625), (237, 5276962), (237, 5291326)],
    "glacier": [(200, 2183923), (276, 6135652), (277, 6157250)],
    "mountain": [(200, 1950345), (262, 5789535), (263, 5849765)],
    "sea": [(200, 1987613), (255, 5655068), (255, 5668011)],
    "street": [(200, 1841817), (250, 5541845), (251, 5547533)],
}

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def class_files(folder):
    return sorted(p.name for p in folder.glob("*.jpg"))

def make_split(data_root):
    rows = []
    rng = np.random.default_rng(SEED)
    for label in CLASSES:
        files = class_files(data_root / FOLDERS["train"] / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        rows += [(files[i], label, "train") for i in idx[:N_TRAIN_PER_CLASS]]
    rng = np.random.default_rng(SEED)
    for label in CLASSES:
        files = class_files(data_root / FOLDERS["test"] / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        rows += [(files[i], label, "validation") for i in idx[:cut]]
        rows += [(files[i], label, "test") for i in idx[cut:]]
    return pd.DataFrame(rows, columns=["file", "label", "part"])

def read_split(path=SPLIT_CSV):
    return pd.read_csv(path)

def control_table(split):
    number = split["file"].str.removesuffix(".jpg").astype(int)
    return split.assign(number=number).groupby(["label", "part"])["number"].agg(["count", "sum"])

def cell(count, total):
    return f"{count} / {total:,}".replace(",", " ")

def print_control(split):
    table = control_table(split)
    print(f"{'Клас':10} {'train':>18} {'validation':>18} {'test':>18} {'Збіг':>6}")
    ok = True
    for label in CLASSES:
        got = [tuple(int(v) for v in table.loc[(label, part)]) for part in PARTS]
        match = got == EXPECTED[label]
        ok &= match
        print(f"{label:10} " + " ".join(f"{cell(*g):>18}" for g in got) + f" {'так' if match else 'ні':>6}")
    totals = [cell(*table.xs(part, level="part").sum()) for part in PARTS]
    print(f"{'Разом':10} " + " ".join(f"{t:>18}" for t in totals))
    repeated = int(split["file"].duplicated().sum())
    print(f"\nПовторів імен у split.csv: {repeated}, тож жоден файл не входить у дві частини: "
          f"{'так' if repeated == 0 else 'НІ'}")
    return ok and repeated == 0

class SplitDataset(Dataset):
    def __init__(self, data_root, split, part, transform):
        rows = split[split["part"] == part]
        self.paths = [data_root / FOLDERS[part] / label / file for file, label in zip(rows["file"], rows["label"])]
        self.labels = [CLASSES.index(label) for label in rows["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(pil_loader(self.paths[i])), self.labels[i]

def make_loader(data_root, split, part):
    if part == "train":
        return DataLoader(SplitDataset(data_root, split, part, train_tf), batch_size=32, shuffle=True,
                          drop_last=False, num_workers=0)
    return DataLoader(SplitDataset(data_root, split, part, eval_tf), batch_size=64, shuffle=False, num_workers=0)


def report(data_root, path=SPLIT_CSV):
    make_split(data_root).to_csv(path, index=False)
    split = read_split(path)
    print(f"Поділ збережено в {path}: {len(split)} рядків\n")
    ok = print_control(split)
    print(f"Усі 18 клітинок збігаються з контрольною таблицею: {'так' if ok else 'НІ'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    args = parser.parse_args()
    report(args.data)

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from torch import nn
import checks
import data
from data import CLASSES, IMG_SIZE, SPLIT_CSV, make_loader, read_split
from model import FROZEN, build_model, get_device, set_train_mode

EPOCHS = 8
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

def run_name(regime, seed):
    return f"{regime}_seed{seed}"

def file_hash(path=SPLIT_CSV):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"],
                               capture_output=True, text=True, check=True).stdout
        return commit, bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None

def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()

def macro_f1(logits, labels):
    return f1_score(labels, logits.argmax(axis=1), average="macro")

def settings(regime, seed, model, optimizer, loaders):
    train_loader, val_loader = loaders
    return {
        "regime": regime,
        "seed": seed,
        "transforms": {"train": repr(data.train_tf), "validation_test": repr(data.eval_tf)},
        "model": {
            "architecture": "torchvision.models.resnet18",
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "head": repr(model.fc),
            "eval_during_training": FROZEN[regime],
            "batch_norm": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
        },
        "optimizer": {
            "type": type(optimizer).__name__,
            "defaults": {k: list(v) if isinstance(v, tuple) else v for k, v in optimizer.defaults.items()},
            "groups": [{"params": sum(p.numel() for p in g["params"]), "lr": g["lr"],
                        "weight_decay": g["weight_decay"]} for g in optimizer.param_groups],
        },
        "loop": {
            "epochs": EPOCHS,
            "steps": EPOCHS * len(train_loader),
            "loss": repr(nn.CrossEntropyLoss()),
            "train_loader": {"batch_size": train_loader.batch_size, "shuffle": True,
                             "drop_last": train_loader.drop_last, "num_workers": train_loader.num_workers},
            "eval_loader": {"batch_size": val_loader.batch_size, "shuffle": False,
                            "num_workers": val_loader.num_workers},
        },
    }

def environment(device):
    commit, dirty = git_state()
    return {
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "split_sha256": file_hash(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }

def train_run(regime, seed, data_root, split, device, out):
    model, optimizer = build_model(regime, seed, device)
    train_loader = make_loader(data_root, split, "train")
    val_loader = make_loader(data_root, split, "validation")
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_macro_f1": []}
    train_time = 0.0
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        set_train_mode(model, regime)
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        train_time += time.perf_counter() - start
        val_logits, val_labels = predict(model, val_loader, device)
        history["train_loss"].append(total / len(train_loader.dataset))
        history["val_macro_f1"].append(macro_f1(val_logits, val_labels))
        print(f"  епоха {epoch}: втрата train {history['train_loss'][-1]:.4f}, "
              f"macro-F1 validation {history['val_macro_f1'][-1]:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    log = {**settings(regime, seed, model, optimizer, (train_loader, val_loader)), **environment(device),
           **history, "train_time_s": train_time}
    (out / "run.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    np.save(out / "val_logits.npy", val_logits)
    torch.save(model.state_dict(), out / "model.pt")
    return log

def select(logs):
    scores = {}
    for log in logs:
        scores.setdefault(log["regime"], []).append((log["val_macro_f1"][-1], log["seed"]))
    regime = max(scores, key=lambda r: np.mean([f for f, _ in scores[r]]))
    seed = max(scores[regime])[1]
    return regime, seed

def save_selected(regime, seed, runs, path):
    state_dict = torch.load(runs / run_name(regime, seed) / "model.pt", map_location="cpu")
    torch.save({"state_dict": state_dict, "regime": regime, "seed": seed,
                "classes": CLASSES, "img_size": IMG_SIZE}, path)


def report(data_root, runs, model_path):
    device = get_device()
    print(f"Пристрій: {device}")
    print("\n\n1. ПЕРЕВІРКИ ДО НАВЧАННЯ\n")
    if not data.report(data_root):
        raise SystemExit("Поділ не збігається з контрольною таблицею, навчання не запускається.")
    split = read_split()
    print()
    if not checks.report(data_root, split, device):
        raise SystemExit("Перевірки режимів не пройдено, навчання не запускається.")
    print("\n\n2. СІМ ЗАПУСКІВ\n")
    logs = []
    for i, (regime, seed) in enumerate(RUNS, 1):
        print(f"Запуск {i}: {regime}, зерно {seed}")
        logs.append(train_run(regime, seed, data_root, split, device, runs / run_name(regime, seed)))
        print(f"  час навчання {logs[-1]['train_time_s']:.1f} с\n")
    print(f"Журнали, логіти validation і ваги збережено в {runs}/")
    print("\n\n3. ВИБІР КОНФІГУРАЦІЇ ЗА VALIDATION\n")
    regime, seed = select(logs)
    save_selected(regime, seed, runs, model_path)
    print(f"Обрано режим {regime}, зерно {seed}. Ваги збережено в {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Тека для журналів запусків")
    parser.add_argument("--model", type=Path, default=Path("lab3_model.pt"), help="Файл вибраної моделі")
    args = parser.parse_args()
    if args.runs.exists() and any(args.runs.iterdir()):
        sys.exit(f"Тека {args.runs} не порожня. Для повторного навчання вкажіть іншу, наприклад --runs runs_rerun.")
    report(args.data, args.runs, args.model)

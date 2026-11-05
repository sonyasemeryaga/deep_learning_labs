import argparse
import math
from itertools import islice
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, make_loader, read_split
from model import FROZEN, TOP_MODULES, build_model, get_device, set_train_mode

N_STEPS = 5
BN_LAYER = "layer1.0.bn1"

def show_batch(data_root, split, path):
    torch.manual_seed(1)
    x, y = next(iter(make_loader(data_root, split, "train")))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.6))
    for ax, image, label in zip(axes.flat, x[:16], y[:16]):
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після перетворень", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def initial_loss(regime, data_root, split, device):
    model, _ = build_model(regime, 1, device)
    x, y = next(iter(make_loader(data_root, split, "train")))
    set_train_mode(model, regime)
    with torch.no_grad():
        logits = model(x.to(device))
        loss = nn.CrossEntropyLoss()(logits, y.to(device))
    return loss.item(), logits.std().item()

def run_steps(regime, data_root, split, device, correct_mode=True):
    model, optimizer = build_model(regime, 1, device)
    loader = make_loader(data_root, split, "train")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if correct_mode:
        set_train_mode(model, regime)
    else:
        model.train()
    loss_fn = nn.CrossEntropyLoss()
    for x, y in islice(loader, N_STEPS):
        optimizer.zero_grad()
        loss_fn(model(x.to(device)), y.to(device)).backward()
        optimizer.step()
    return model, before, model.state_dict()

def changed(before, after, prefix, names):
    return any(not torch.equal(before[f"{prefix}.{k}"], after[f"{prefix}.{k}"]) for k in names)

def change_table(model, before, after, regime):
    rows = []
    for name in TOP_MODULES:
        module = getattr(model, name)
        params = changed(before, after, name, [k for k, _ in module.named_parameters()])
        buffers = [k for k, _ in module.named_buffers()]
        stats = changed(before, after, name, buffers) if buffers else None
        trainable = name not in FROZEN[regime]
        expected = (trainable, trainable if buffers else None)
        rows.append((name, params, stats, (params, stats) == expected))
    return rows

def yes_no(value):
    return "немає BN" if value is None else "так" if value else "ні"

def print_changes(regime, rows):
    print(f"{'Режим':9} {'Модуль':7} {'Параметри змінилися':>20} {'Статистики BN змінилися':>24} {'Збіг':>5}")
    for name, params, stats, match in rows:
        print(f"{regime:9} {name:7} {yes_no(params):>20} {yes_no(stats):>24} {yes_no(match):>5}")

def bn_shift(before, after):
    return {stat: (after[f"{BN_LAYER}.{stat}"] - before[f"{BN_LAYER}.{stat}"]).abs().max().item()
            for stat in ["running_mean", "running_var"]}


def report(data_root, split, device, out=Path("figures")):
    show_batch(data_root, split, out / "train_batch.png")
    print(f"Рисунок пакета після перетворень збережено в {out}/train_batch.png\n")
    print(f"Початкова втрата на першому пакеті, зерно 1, ln 6 = {math.log(6):.4f}:")
    print(f"{'Режим':9} {'L0':>8} {'L0 - ln 6':>10} {'std логітів':>12}")
    for regime in ["scratch", "frozen"]:
        loss, spread = initial_loss(regime, data_root, split, device)
        print(f"{regime:9} {loss:>8.4f} {loss - math.log(6):>+10.4f} {spread:>12.4f}")
    print(f"\nЩо змінюється за {N_STEPS} кроків оптимізатора з set_train_mode:")
    ok = True
    for regime in ["frozen", "unfreeze"]:
        model, before, after = run_steps(regime, data_root, split, device)
        rows = change_table(model, before, after, regime)
        ok &= all(row[-1] for row in rows)
        print_changes(regime, rows)
        print()
    print(f"Усі модулі поводяться як очікувано: {'так' if ok else 'НІ'}")
    print(f"\nДослід із навмисною помилкою: frozen із model.train() замість set_train_mode")
    model, before, after = run_steps("frozen", data_root, split, device, correct_mode=False)
    print_changes("frozen", change_table(model, before, after, "frozen"))
    shift = bn_shift(before, after)
    print(f"\nМаксимальна абсолютна зміна в {BN_LAYER} після {N_STEPS} кроків:")
    print(f"  running_mean: {shift['running_mean']:.6g}")
    print(f"  running_var:  {shift['running_var']:.6g}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    args = parser.parse_args()
    device = get_device()
    print(f"Пристрій: {device}\n")
    report(args.data, read_split(), device)

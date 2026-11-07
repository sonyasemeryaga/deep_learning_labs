import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torchvision.datasets.folder import pil_loader
from torchvision.models import resnet18
from data import CLASSES, FOLDERS, make_loader, read_split
from train import RUNS, file_hash, predict, run_name, select

COLORS = {"scratch": "#52514e", "frozen": "#2a78d6", "unfreeze": "#eb6834"}
STYLES = {1: "-", 2: "--", 3: ":"}
N_ERRORS = 5

def load_logs(runs):
    logs = [json.loads((runs / run_name(regime, seed) / "run.json").read_text()) for regime, seed in RUNS]
    current = file_hash()
    hashes = {log["split_sha256"] for log in logs}
    if hashes != {current}:
        raise SystemExit(f"Хеш split.csv у журналах ({', '.join(sorted(hashes))}) не збігається "
                         f"з поточним ({current}). Аналіз зупинено.")
    print(f"Хеш split.csv однаковий у всіх {len(logs)} журналах і збігається з поточним: {current}")
    return logs

def runs_table(logs):
    return pd.DataFrame([{"#": i, "Режим": log["regime"], "Зерно": log["seed"],
                          "Втрата train, епоха 8": log["train_loss"][-1],
                          "macro-F1 validation, епоха 8": log["val_macro_f1"][-1],
                          "Час навчання, с": log["train_time_s"]} for i, log in enumerate(logs, 1)])

def print_runs(table):
    print(f"{'#':>2} {'Режим':9} {'Зерно':>5} {'Втрата train':>13} {'macro-F1 val':>13} {'Час, с':>8}")
    for _, r in table.iterrows():
        print(f"{r['#']:>2} {r['Режим']:9} {r['Зерно']:>5} {r['Втрата train, епоха 8']:>13.4f}"
              f" {r['macro-F1 validation, епоха 8']:>13.4f} {r['Час навчання, с']:>8.1f}")

def style(ax, title, ylabel):
    ax.set_title(title, loc="center", fontsize=12)
    ax.set_xlabel("Епоха")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

def plot_curves(logs, path):
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.8))
    for log in logs:
        epochs = range(1, len(log["train_loss"]) + 1)
        kw = {"color": COLORS[log["regime"]], "linestyle": STYLES[log["seed"]], "linewidth": 2,
              "marker": "o", "markersize": 4, "label": f"{log['regime']}, зерно {log['seed']}"}
        ax_loss.plot(epochs, log["train_loss"], **kw)
        ax_f1.plot(epochs, log["val_macro_f1"], **kw)
    style(ax_loss, "Втрата на train", "Втрата")
    style(ax_f1, "macro-F1 на validation", "macro-F1")
    ax_loss.set_ylim(bottom=0)
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def effects(logs):
    scores = {}
    for log in logs:
        scores.setdefault(log["regime"], []).append(log["val_macro_f1"][-1])
    mean = {r: float(np.mean(f)) for r, f in scores.items()}
    spread = {r: float(np.ptp(f)) for r, f in scores.items() if len(f) > 1}
    noise = max(spread.values())
    return {"F_scratch_1": mean["scratch"], "F_frozen": mean["frozen"], "R_frozen": spread["frozen"],
            "F_unfreeze": mean["unfreeze"], "R_unfreeze": spread["unfreeze"], "R": noise,
            "delta_pre": mean["frozen"] - mean["scratch"], "delta_unf": mean["unfreeze"] - mean["frozen"]}

def verdict(delta, noise):
    if abs(delta) > noise:
        return "|Δ| > R, ефект вартий перевірки, але не доведений"
    return "|Δ| ≤ R, різниця не відрізняється від розкиду, результату немає"

def print_effects(e):
    rows = [("F_frozen, R_frozen", f"{e['F_frozen']:.4f}, {e['R_frozen']:.4f}"),
            ("F_unfreeze, R_unfreeze", f"{e['F_unfreeze']:.4f}, {e['R_unfreeze']:.4f}"),
            ("R", f"{e['R']:.4f}"),
            ("Δpre", f"{e['delta_pre']:+.4f}"),
            ("Δunf", f"{e['delta_unf']:+.4f}"),
            ("Δpre + Δunf", f"{e['delta_pre'] + e['delta_unf']:+.4f}")]
    print(f"{'Величина':24} {'Значення':>16}")
    for name, value in rows:
        print(f"{name:24} {value:>16}")
    print(f"\nΔpre: {verdict(e['delta_pre'], e['R'])}")
    print(f"Δunf: {verdict(e['delta_unf'], e['R'])}")

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = resnet18(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint

def check_selection(logs, model, checkpoint, data_root, split, runs):
    regime, seed = select(logs)
    print(f"За журналами правило вибору дає: {regime}, зерно {seed}")
    print(f"У lab3_model.pt: {checkpoint['regime']}, зерно {checkpoint['seed']}, "
          f"класи {checkpoint['classes']}, img_size {checkpoint['img_size']}")
    if (checkpoint["regime"], checkpoint["seed"]) != (regime, seed):
        raise SystemExit("lab3_model.pt не відповідає правилу вибору. Тестову оцінку не виконано.")
    saved = np.load(runs / run_name(regime, seed) / "val_logits.npy")
    logits, labels = predict(model, make_loader(data_root, split, "validation"), torch.device("cpu"))
    agree = float(np.mean(saved.argmax(axis=1) == logits.argmax(axis=1)))
    print(f"Логіти validation цієї моделі на CPU проти збережених у журналі: "
          f"найбільша різниця {np.abs(saved - logits).max():.2e}, збіг передбачень {agree:.4f}")
    return regime, seed

def test_metrics(logits, labels):
    p = accuracy_score(labels, logits.argmax(axis=1))
    return {"macro_f1": f1_score(labels, logits.argmax(axis=1), average="macro"), "accuracy": p,
            "n": len(labels), "se_accuracy": math.sqrt(p * (1 - p) / len(labels))}

def print_confusion(cm):
    print(f"{'справжній \\ передбачений':>24} " + " ".join(f"{c:>10}" for c in CLASSES))
    for name, row in zip(CLASSES, cm):
        print(f"{name:>24} " + " ".join(f"{v:>10}" for v in row))

def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Матриця плутанини на test", loc="center", fontsize=12)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def characteristic_errors(cm, files, labels, logits):
    pred = logits.argmax(axis=1)
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pairs = sorted(((cm[t, p], t, p) for t in range(len(CLASSES)) for p in range(len(CLASSES))
                    if t != p and cm[t, p] > 0), key=lambda x: (-x[0], x[1], x[2]))
    candidates = [sorted((files[i], i) for i in np.flatnonzero((labels == t) & (pred == p))) for _, t, p in pairs]
    rows = []
    for k in range(max((len(c) for c in candidates), default=0)):
        for (count, t, p), files_of_pair in zip(pairs, candidates):
            if k < len(files_of_pair) and len(rows) < N_ERRORS:
                file, i = files_of_pair[k]
                rows.append({"Файл": file, "Справжній клас": CLASSES[t], "Передбачений клас": CLASSES[p],
                             "Імовірність передбаченого класу": float(prob[i, p]),
                             "Кількість таких помилок у test": int(count)})
    return pd.DataFrame(rows)

def print_errors(errors):
    print(f"{'Файл':>10} {'Справжній':>10} {'Передбачений':>13} {'Імовірність':>12} {'Кількість':>10}")
    for _, e in errors.iterrows():
        print(f"{e['Файл']:>10} {e['Справжній клас']:>10} {e['Передбачений клас']:>13}"
              f" {e['Імовірність передбаченого класу']:>12.4f} {e['Кількість таких помилок у test']:>10}")

def plot_errors(errors, data_root, path):
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 4.3), layout="constrained")
    for ax, (_, e) in zip(np.atleast_1d(axes), errors.iterrows()):
        image = pil_loader(data_root / FOLDERS["test"] / e["Справжній клас"] / e["Файл"])
        ax.imshow(image)
        probability = f"{e['Імовірність передбаченого класу']:.2f}".replace(".", ",")
        ax.set_title(f"{e['Файл']}\n{e['Справжній клас']} → {e['Передбачений клас']}\np = {probability}",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle("П'ять характерних помилок на test (справжній → передбачений клас)", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def section(number, title):
    print(f"\n\n{number}. {title}\n")


def report(data_root, runs, model_path, out):
    print("Лабораторна робота №3")
    print("з дисципліни «Системи глибинного навчання»")
    print("студентки групи КМ-31")
    print("Семеряги Софії")
    print("\nОбчислювальний експеримент і трансферне навчання")
    out.mkdir(parents=True, exist_ok=True)
    split = read_split()
    section(1, "ЖУРНАЛИ СЕМИ ЗАПУСКІВ")
    logs = load_logs(runs)
    commits = {(log["git_commit"], log["git_uncommitted_changes"]) for log in logs}
    print(f"Коміт коду й незакомічені зміни в журналах: {', '.join(f'{c} ({d})' for c, d in sorted(commits))}\n")
    table = runs_table(logs)
    print_runs(table)
    table.to_csv(out / "runs_table.csv", index=False)
    plot_curves(logs, out / "curves.png")
    print(f"\nТаблицю збережено в {out}/runs_table.csv, криві в {out}/curves.png")
    section(2, "РОЗКЛАД ЕФЕКТІВ")
    e = effects(logs)
    print_effects(e)
    (out / "effects.json").write_text(json.dumps(e, ensure_ascii=False, indent=2) + "\n")
    print(f"\nЕфекти збережено в {out}/effects.json")
    section(3, "ПЕРЕВІРКА ВИБОРУ КОНФІГУРАЦІЇ")
    model, checkpoint = load_model(model_path)
    regime, seed = check_selection(logs, model, checkpoint, data_root, split, runs)
    f_selected = next(log["val_macro_f1"][-1] for log in logs if (log["regime"], log["seed"]) == (regime, seed))
    if abs(e["delta_unf"]) <= e["R"]:
        print("|Δunf| ≤ R, тому вибір між frozen і unfreeze визначив шум.")
    section(4, "ТЕСТОВА ОЦІНКА НА CPU")
    logits, labels = predict(model, make_loader(data_root, split, "test"), torch.device("cpu"))
    m = test_metrics(logits, labels)
    print(f"Test: {m['n']} зображень")
    print(f"macro-F1 на test:              {m['macro_f1']:.4f}")
    print(f"macro-F1 на validation (F_r,s): {f_selected:.4f}")
    print(f"Різниця test - validation:      {m['macro_f1'] - f_selected:+.4f}")
    print(f"accuracy на test p = {m['accuracy']:.4f}, sqrt(p(1 - p) / n) = {m['se_accuracy']:.4f}\n")
    cm = confusion_matrix(labels, logits.argmax(axis=1), labels=range(len(CLASSES)))
    print("Матриця плутанини (рядки — справжній клас, стовпці — передбачений):")
    print_confusion(cm)
    plot_confusion(cm, out / "confusion_matrix.png")
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "confusion_matrix.csv")
    files = split.loc[split["part"] == "test", "file"].to_numpy()
    errors = characteristic_errors(cm, files, labels, logits)
    print("\nП'ять характерних помилок:")
    print_errors(errors)
    errors.to_csv(out / "test_errors.csv", index=False)
    plot_errors(errors, data_root, out / "test_errors.png")
    (out / "test_metrics.json").write_text(json.dumps({"regime": regime, "seed": seed, "val_macro_f1": f_selected,
                                                       **m}, ensure_ascii=False, indent=2) + "\n")
    print(f"\nРезультати збережено в {out}/: test_metrics.json, confusion_matrix.csv і .png, "
          f"test_errors.csv і .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Тека з журналами запусків")
    parser.add_argument("--model", type=Path, default=Path("lab3_model.pt"), help="Файл вибраної моделі")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    report(args.data, args.runs, args.model, args.out)

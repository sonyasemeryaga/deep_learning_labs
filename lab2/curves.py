import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from metrics import confusion, cost, fmt

CURVE = "#2a78d6"
T_STAR = "#eb6834"
T_HALF = "#52514e"

def operating_point(y, s, t):
    tp, fp, fn, tn = confusion(y, s, t)
    return {"TPR": tp / (tp + fn), "FPR": fp / (fp + tn),
            "precision": tp / (tp + fp) if tp + fp > 0 else None,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="center", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

def mark(ax, x, y, color, text, offset):
    ax.plot(x, y, "o", markersize=8, color=color, markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=10, color="#0b0b0b")

def plot_pr(y, s, t_star, path):
    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(recall, precision, color=CURVE, linewidth=2, drawstyle="steps-post")
    for t, color, name, offset in [(0.5, T_HALF, "t = 0,5", (-22, 12)), (t_star, T_STAR, "t*", (-24, -4))]:
        p = operating_point(y, s, t)
        mark(ax, p["TPR"], p["precision"], color, name, offset)
    style(ax, "Крива точність–повнота (validation)", "Повнота (recall)", "Точність (precision)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_roc(y, s, t_star, auc, path):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(fpr, tpr, color=CURVE, linewidth=2)
    for t, color, name, offset in [(0.5, T_HALF, "t = 0,5", (10, -14)), (t_star, T_STAR, "t*", (8, -16))]:
        p = operating_point(y, s, t)
        mark(ax, p["FPR"], p["TPR"], color, name, offset)
    style(ax, f"ROC-крива (validation), ROC-AUC = {auc:.4f}".replace(".", ","),
          "Частка хибних тривог FPR (логарифмічна шкала, біля нуля лінійна)", "Повнота TPR")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(y, s, t_star, out):
    auc = roc_auc_score(y, s)
    print(f"ROC-AUC на validation: {auc:.6f}\n")
    print(f"{'Поріг':>12} {'TPR':>7} {'FPR':>10} {'precision':>10} {'FP':>5} {'C':>5}")
    fpr, tpr, thr = roc_curve(y, s)
    t90 = thr[np.argmax(tpr >= 0.9)]
    for name, t in [("0.5", 0.5), ("t*", t_star), ("TPR >= 0.9", t90)]:
        p = operating_point(y, s, t)
        print(f"{name:>12} {p['TPR']:>7.4f} {p['FPR']:>10.6f} {fmt(p['precision']):>10}"
              f" {p['FP']:>5} {p['C']:>5}")
    out.mkdir(exist_ok=True)
    plot_pr(y, s, t_star, out / "pr_curve_validation.png")
    plot_roc(y, s, t_star, auc, out / "roc_curve_validation.png")
    print(f"\nКриві збережено в {out}/pr_curve_validation.png і {out}/roc_curve_validation.png")
    return auc


if __name__ == "__main__":
    import json
    from pathlib import Path
    from lab2_starter import load_predictions
    data = load_predictions(Path("results"))
    t_star = json.loads(Path("analysis/threshold.json").read_text())["t_star"]
    report(data["validation_y"], data["validation_scores"], t_star, Path("analysis"))

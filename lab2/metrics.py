import numpy as np

COST_FN = 10
COST_FP = 1

def confusion(y, s, t):
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn

def cost(fn, fp):
    return COST_FN * fn + COST_FP * fp

def precision(tp, fp):
    return tp / (tp + fp) if tp + fp > 0 else None

def recall(tp, fn):
    return tp / (tp + fn) if tp + fn > 0 else None

def fmt(value):
    return "не визначено" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    from pathlib import Path
    from lab2_starter import load_predictions
    data = load_predictions(Path("results"))
    y = data["validation_y"]
    s = data["validation_scores"]
    print(f"Validation: {len(y)} транзакцій, з них {int(y.sum())} шахрайських")
    print(f"Правило: шахрайство, якщо s >= t, C = {COST_FN} * FN + {COST_FP} * FP\n")
    print(f"{'Поріг':>8} {'TP':>5} {'FP':>6} {'FN':>5} {'TN':>7} {'C':>6}  {'precision':>12} {'recall':>12}")
    for t in [0.5, 1 / 11, np.inf]:
        tp, fp, fn, tn = confusion(y, s, t)
        print(f"{t:>8.4f} {tp:>5} {fp:>6} {fn:>5} {tn:>7} {cost(fn, fp):>6}"
              f"  {fmt(precision(tp, fp)):>12} {fmt(recall(tp, fn)):>12}")

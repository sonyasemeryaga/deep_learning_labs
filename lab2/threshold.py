import json
import numpy as np
import pandas as pd
from metrics import COST_FN, COST_FP, confusion, cost

def candidate_table(y, s):
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    last = np.append(s_sorted[1:] != s_sorted[:-1], True)
    thresholds = np.append(np.float32(np.inf), s_sorted[last])
    tp = np.append(0, tp[last])
    fp = np.append(0, fp[last])
    P = int(np.sum(y == 1))
    N = len(y) - P
    fn = P - tp
    tn = N - fp
    c = COST_FN * fn + COST_FP * fp
    return {"threshold": thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c}

def best_threshold(table):
    c = table["C"]
    return float(table["threshold"][c == c.min()].max())

def print_neighbourhood(table, t_star, k=3):
    i = int(np.flatnonzero(table["threshold"] == np.float32(t_star))[0])
    print(f"{'Поріг':>12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>7} {'C':>5}")
    for j in range(max(i - k, 0), min(i + k + 1, len(table["threshold"]))):
        mark = "  <- t*" if j == i else ""
        print(f"{table['threshold'][j]:>12.9g} {table['TP'][j]:>4} {table['FP'][j]:>4}"
              f" {table['FN'][j]:>4} {table['TN'][j]:>7} {table['C'][j]:>5}{mark}")

def summary_rows(y, s, t_star):
    rows = []
    for rule, t in [("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", np.inf)]:
        tp, fp, fn, tn = confusion(y, s, t)
        rows.append((rule, t, tp, fp, fn, tn, cost(fn, fp)))
    return rows

def print_summary(rows, split="validation"):
    print(f"{'Правило на ' + split:22} {'Поріг':>12} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>7} {'C':>5}")
    for rule, t, tp, fp, fn, tn, c in rows:
        print(f"{rule:22} {t:>12.9g} {tp:>4} {fp:>4} {fn:>4} {tn:>7} {c:>5}")

def save(table, t_star, out):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out / "candidates_validation.csv", index=False)
    i = int(np.flatnonzero(table["threshold"] == np.float32(t_star))[0])
    fixed = {"t_star": t_star, **{k: int(table[k][i]) for k in ["TP", "FP", "FN", "TN", "C"]}}
    (out / "threshold.json").write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n")


def report(y, s, out):
    table = candidate_table(y, s)
    t_star = best_threshold(table)
    print(f"Validation: {len(s)} оцінок, з них унікальних {len(np.unique(s))}")
    print(f"Кандидатів разом із +inf: {len(table['threshold'])}")
    print(f"Мінімальна вартість C = {table['C'].min()}")
    print(f"Кількість порогів із мінімальною вартістю: {np.sum(table['C'] == table['C'].min())}")
    print(f"Обрано найбільший із них: t* = {t_star:.9g}\n")
    print("Фрагмент повної таблиці кандидатів навколо t*:")
    print_neighbourhood(table, t_star)
    print("\nПідсумкова таблиця:")
    print_summary(summary_rows(y, s, t_star))
    save(table, t_star, out)
    print(f"\nПовну таблицю кандидатів збережено в {out}/candidates_validation.csv")
    print(f"Зафіксований поріг збережено в {out}/threshold.json")
    return t_star


if __name__ == "__main__":
    from pathlib import Path
    from lab2_starter import load_predictions
    data = load_predictions(Path("results"))
    report(data["validation_y"], data["validation_scores"], Path("analysis"))

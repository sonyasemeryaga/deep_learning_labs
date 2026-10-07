import json
import numpy as np
import pandas as pd
from metrics import confusion, cost, precision, recall, fmt

def test_report(y, s, t_star):
    tp, fp, fn, tn = confusion(y, s, t_star)
    return {"t_star": t_star, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp),
            "precision": precision(tp, fp), "recall": recall(tp, fn)}

def first_errors(y, s, amount, row_ids, t_star, k=5):
    pred = s >= t_star
    order = np.argsort(row_ids)
    rows = []
    for kind, mask in [("FP", pred & (y == 0)), ("FN", ~pred & (y == 1))]:
        idx = order[mask[order]][:k]
        for i in idx:
            rows.append({"Рядок CSV": int(row_ids[i]), "Тип помилки": kind, "Мітка": int(y[i]),
                         "Оцінка": float(s[i]), "s - t*": float(s[i]) - t_star,
                         "Amount": float(amount[i])})
    return pd.DataFrame(rows)

def print_test(r):
    print(f"Поріг t* = {r['t_star']:.9g} (зафіксовано на validation)\n")
    print(f"{'':12} {'прогноз 1':>12} {'прогноз 0':>12}")
    print(f"{'справді 1':12} {'TP = ' + str(r['TP']):>12} {'FN = ' + str(r['FN']):>12}")
    print(f"{'справді 0':12} {'FP = ' + str(r['FP']):>12} {'TN = ' + str(r['TN']):>12}")
    print(f"\nC = 10 * {r['FN']} + {r['FP']} = {r['C']}")
    print(f"precision = {r['TP']} / ({r['TP']} + {r['FP']}) = {fmt(r['precision'])}")
    print(f"recall    = {r['TP']} / ({r['TP']} + {r['FN']}) = {fmt(r['recall'])}")

def print_errors(errors):
    print(f"{'Рядок CSV':>10} {'Тип':>4} {'Мітка':>6} {'Оцінка':>12} {'s - t*':>12} {'Amount':>9}")
    for _, e in errors.iterrows():
        print(f"{e['Рядок CSV']:>10} {e['Тип помилки']:>4} {e['Мітка']:>6} {e['Оцінка']:>12.6g}"
              f" {e['s - t*']:>+12.6g} {e['Amount']:>9.2f}")

def save(r, errors, out):
    (out / "test_metrics.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    errors.to_csv(out / "test_errors.csv", index=False)


def report(data, t_star, out):
    from threshold import summary_rows, print_summary
    y, s = data["test_y"], data["test_scores"]
    print(f"Test: {len(y)} транзакцій, з них {int(y.sum())} шахрайських\n")
    r = test_report(y, s, t_star)
    print_test(r)
    rows = summary_rows(y, s, t_star)
    print("\nДля порівняння ті самі правила на test:")
    print_summary(rows, split="test")
    errors = first_errors(y, s, data["test_amount"], data["test_row_ids"], t_star)
    print("\nПерші п'ять FP і перші п'ять FN за номером рядка CSV:")
    print_errors(errors)
    save(r, errors, out)
    print(f"\nРезультати збережено в {out}/test_metrics.json і {out}/test_errors.csv")
    return rows


if __name__ == "__main__":
    from pathlib import Path
    from lab2_starter import load_predictions
    from brute_check import SETS, check
    if not all(check(name)[-1] for name in SETS):
        raise SystemExit("Перевірку пошуку порога на наборах A, B, C не пройдено, test не оцінюється.")
    t_star = json.loads(Path("analysis/threshold.json").read_text())["t_star"]
    report(load_predictions(Path("results")), t_star, Path("analysis"))

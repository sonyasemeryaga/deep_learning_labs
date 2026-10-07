import numpy as np
import pandas as pd
from metrics import confusion, cost
from threshold import candidate_table, best_threshold

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}

def brute_table(y, s):
    candidates = [np.inf] + sorted(np.unique(s).tolist(), reverse=True)
    rows = []
    for t in candidates:
        tp, fp, fn, tn = confusion(y, s, t)
        rows.append((t, tp, fp, fn, tn, cost(fn, fp)))
    return rows

def brute_best(rows):
    best_t, best_c = None, None
    for t, _, _, _, _, c in rows:
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_t, best_c = t, c
    return best_t, best_c

def check(name):
    s = np.array(SETS[name][0])
    y = np.array(SETS[name][1])
    rows = brute_table(y, s)
    fast = candidate_table(y, s)
    fast_rows = list(zip(fast["threshold"].tolist(), fast["TP"].tolist(), fast["FP"].tolist(),
                         fast["FN"].tolist(), fast["TN"].tolist(), fast["C"].tolist()))
    t_brute, c_brute = brute_best(rows)
    t_fast = best_threshold(fast)
    same_rows = rows == fast_rows
    same_t = t_brute == t_fast
    return s, y, rows, same_rows, t_brute, c_brute, t_fast, same_rows and same_t

def report(name):
    s, y, rows, same_rows, t_brute, c_brute, t_fast, passed = check(name)
    print(f"Набір {name}: s = {s.tolist()}, y = {y.tolist()}")
    print(f"  {'Поріг':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'C':>4}")
    for t, tp, fp, fn, tn, c in rows:
        print(f"  {t:>6.1f} {tp:>3} {fp:>3} {fn:>3} {tn:>3} {c:>4}")
    print(f"  Повільний перебір: t* = {t_brute}, C = {c_brute}")
    print(f"  Швидкий алгоритм:  t* = {t_fast}")
    print(f"  Таблиці збігаються: {'так' if same_rows else 'НІ'}, "
          f"перевірку пройдено: {'так' if passed else 'НІ'}\n")
    return passed

def save(out):
    rows = []
    for name in SETS:
        _, _, table, same_rows, t_brute, _, t_fast, passed = check(name)
        for t, tp, fp, fn, tn, c in table:
            rows.append({"set": name, "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c,
                         "selected": t == t_brute, "t_star_fast": t_fast, "passed": passed})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "threshold_checks.csv", index=False)


if __name__ == "__main__":
    results = [report(name) for name in SETS]
    print(f"Усі три перевірки пройдено: {'так' if all(results) else 'НІ'}")

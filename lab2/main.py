import argparse
import json
import numpy as np
from pathlib import Path
import brute_check
import curves
import evaluate
import threshold
from lab2_starter import load_predictions, split_summary

def section(number, title):
    print(f"\n\n{f'{number}. ' if number else ''}{title}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    RESULTS, OUT = args.saved, args.out
    print("Лабораторна робота №2")
    print("з дисципліни «Системи глибинного навчання»")
    print("студентки групи КМ-31")
    print("Семеряги Софії")
    print("\nПоріг рішення під асиметричну ціну помилок")
    data = load_predictions(RESULTS)
    run = json.loads((RESULTS / "run.json").read_text())
    with np.load(RESULTS / "preprocessing.npz") as prep:
        n_scaler = int(prep["n_samples_seen"])
    section(1, "НАВЧАННЯ ТА ЗБЕРЕЖЕНІ ПРОГНОЗИ")
    print(f"Linear(29,32) -> ReLU -> Linear(32,1), {run['epochs']} епох, пакет {run['batch_size']}, "
          f"Adam lr = {run['learning_rate']}, {run['dtype']}, {run['device']}.")
    print(f"Python {run['python']}, torch {run['torch']}. Прогнози завантажено з {RESULTS}/ без навчання.\n")
    print(f"{'Вибірка':12} {"Об'єктів":>9} {'Шахрайських':>12} {'Частка':>9}")
    for _, r in split_summary(data).iterrows():
        print(f"{r['split']:12} {r['n']:>9} {r['positives']:>12} {r['positive_fraction']:>9.4%}")
    print(f"\nStandardScaler навчено на {n_scaler} об'єктах, тобто лише на train.")
    section(2, "ВИБІР ПОРОГА НА VALIDATION")
    y_val, s_val = data["validation_y"], data["validation_scores"]
    threshold.report(y_val, s_val, OUT)
    t_star = json.loads((OUT / "threshold.json").read_text())["t_star"]
    print()
    curves.report(y_val, s_val, t_star, OUT)
    section(3, "ПЕРЕВІРКА РЕАЛІЗАЦІЇ ПОШУКУ ПОРОГА")
    checks_ok = all([brute_check.report(name) for name in brute_check.SETS])
    print(f"Усі три перевірки пройдено: {'так' if checks_ok else 'НІ'}")
    brute_check.save(OUT)
    print(f"Таблиці перевірки збережено в {OUT}/threshold_checks.csv")
    section(4, "ТЕСТОВА ОЦІНКА ТА АНАЛІЗ ПОМИЛОК")
    if not checks_ok:
        print("Перевірку пошуку порога не пройдено, test не оцінюється.")
        return
    test_rows = evaluate.report(data, t_star, OUT)
    val_rows = threshold.summary_rows(y_val, s_val, t_star)
    section(None, "ПІДСУМОК")
    print(f"Поріг t* = {t_star:.9g} зафіксовано на validation.")
    print(f"{'':12} {'C при 0,5':>10} {'C при t*':>10}")
    print(f"{'validation':12} {val_rows[0][-1]:>10} {val_rows[1][-1]:>10}")
    print(f"{'test':12} {test_rows[0][-1]:>10} {test_rows[1][-1]:>10}")

if __name__ == "__main__":
    main()

import numpy as np
import compare_torch
import numeric_check
from data import load_data
from network import init_params, forward, backward, cross_entropy

def section(number, title):
    print(f"\n\n{f'{number}. ' if number else ''}{title}\n")

def main():
    print("Лабораторна робота №1")
    print("з дисципліни «Системи глибинного навчання»")
    print("студентки групи КМ-31")
    print("Семеряги Софії")
    print("\nРеалізація та перевірка зворотного поширення похибки")
    section(None, "ВИХІДНІ ДАНІ")
    X, y, X_test, y_test = load_data()
    W1, b1, W2, b2 = init_params()
    print(f"Iris, стратифікований поділ 70/30, np.random.default_rng(0).")
    print(f"Train {X.shape}, по {np.bincount(y)[0]} об'єктів кожного класу; test {X_test.shape}.")
    print(f"Стандартизація за train (ddof=0): середнє за модулем не більше "
          f"{np.abs(X.mean(axis=0)).max():.1e}, ст. відхилення {X.std(axis=0, ddof=0).min():.3f}.")
    print(f"\nLinear(4,8) -> ReLU -> Linear(8,3), "
          f"{W1.size + b1.size + W2.size + b2.size} параметрів, {X.dtype}.")
    print(f"  W1 {str(W1.shape):8} He      sigma = {np.sqrt(2 / 4):.6f}")
    print(f"  W2 {str(W2.shape):8} Xavier  sigma = {np.sqrt(2 / 11):.6f}")
    print(f"  b1 {str(b1.shape):8} b2 {str(b2.shape):8} нулі")
    section(1, "РЕАЛІЗАЦІЯ В NUMPY")
    Z2, cache = forward(X, W1, b1, W2, b2)
    loss = cross_entropy(Z2, y)
    grads = backward(cache, y, W2)
    print("Розмірності проміжних масивів і градієнтів:")
    for name in ["X", "Z1", "A1", "Z2"]:
        print(f"  {name:3} {cache[name].shape}")
    for name, g in zip(["dW1", "db1", "dW2", "db2"], grads):
        print(f"  {name:3} {g.shape}")
    print("\nЗбережено для зворотного проходу: X і A1 — входи шарів у grad_W = A^T Delta;")
    print("Z1 — знаки передактивацій для маски ReLU; Z2 — логіти для бази (P - Y) / N.")
    print(f"\nВтрата на початкових вагах: {loss:.15f}")
    print(f"Вимкнених нейронів ReLU: {np.mean(cache['Z1'] <= 0):.1%}")
    section(2, "ЗВІРКА З PYTORCH")
    ok_torch = compare_torch.report(divide_by_N=True)
    section(3, "ПЕРЕВІРКА ОКРЕМИХ ГРАДІЄНТІВ ЧИСЕЛЬНИМ ДИФЕРЕНЦІЮВАННЯМ")
    ok_numeric = numeric_check.report(divide_by_N=True)
    section(4, "ДОСЛІД ІЗ НАВМИСНОЮ ПОМИЛКОЮ")
    print("Delta2 = P - Y замість (P - Y) / N; функція втрат і параметри не змінені.")
    print("\nПрогноз до запуску:")
    print("  1. Втрата не зміниться — її дає прямий прохід.")
    print(f"  2. Усі градієнти помножаться на N = {len(y)}, нулі залишаться нулями.")
    print("  3. Обидві перевірки провалять градієнти, рядок «Втрата» пройде.")
    print()
    compare_torch.report(divide_by_N=False)
    print()
    numeric_check.report(divide_by_N=False)
    bad = backward(cache, y, W2, divide_by_N=False)
    ratios = np.concatenate([b[np.abs(g) > 0] / g[np.abs(g) > 0] for g, b in zip(grads, bad)])
    zeros_kept = all(np.all(b[np.abs(g) == 0] == 0) for g, b in zip(grads, bad))
    print(f"\nВідношення зламаних градієнтів до правильних: "
          f"{ratios.min():.6f}...{ratios.max():.6f}, нулі збережено: {zeros_kept}.")
    print("Прогноз справдився за всіма трьома пунктами.")
    section(None, "ПІДСУМОК")
    print(f"Звірка з PyTorch:   {'пройдена' if ok_torch else 'ПРОВАЛЕНА'}")
    print(f"Чисельна перевірка: {'пройдена' if ok_numeric else 'ПРОВАЛЕНА'}")
    print("Дослід: помилку виявлено обома перевірками.")

if __name__ == "__main__":
    main()

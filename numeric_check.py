import numpy as np
from data import load_data
from network import init_params, forward, backward, loss_from_params

EPS = 1e-6
TOL = 1e-7

def central_difference(X, y, params, which, index):
    theta = params[which]
    original = theta[index]
    theta[index] = original + EPS
    loss_plus = loss_from_params(X, y, *params)
    theta[index] = original - EPS
    loss_minus = loss_from_params(X, y, *params)
    theta[index] = original
    return (loss_plus - loss_minus) / (2 * EPS)


def relu_mask_is_stable(X, y, params, which, index):
    theta = params[which]
    original = theta[index]
    masks = []
    for shift in (-EPS, 0.0, EPS):
        theta[index] = original + shift
        _, cache = forward(X, *params)
        masks.append(cache["Z1"] > 0)
    theta[index] = original
    return np.array_equal(masks[0], masks[1]) and np.array_equal(masks[1], masks[2])

def run(divide_by_N=True):
    X, y, _, _ = load_data()
    params = list(init_params())
    _, cache = forward(X, *params)
    grads = backward(cache, y, params[2], divide_by_N=divide_by_N)
    checks = [("W1[0,0]", 0, (0, 0)),
              ("b1[0]",   1, (0,)),
              ("W2[0,0]", 2, (0, 0)),
              ("b2[0]",   3, (0,))]
    rows = []
    for label, which, index in checks:
        g_manual = grads[which][index]
        g_num = central_difference(X, y, params, which, index)
        stable = relu_mask_is_stable(X, y, params, which, index)
        rows.append((label, g_manual, g_num, abs(g_num - g_manual), stable))
    return rows

def report(divide_by_N=True):
    rows = run(divide_by_N)
    print(f"Центральна різниця, крок eps = {EPS:.0e}, допуск {TOL:.0e}\n")
    print(f"{'Параметр':10} {'Градієнт backward()':>22} {'Чисельна похідна':>22}"
          f" {'Абс. різниця':>14}  Пройдено")
    for label, g_manual, g_num, diff, _ in rows:
        print(f"{label:10} {g_manual:>22.15f} {g_num:>22.15f}"
              f" {diff:>14.3e}  {'так' if diff <= TOL else 'НІ'}")
    if all(stable for *_, stable in rows):
        print("\nЗбурення не перетинає злам ReLU: маски при theta-eps, theta, theta+eps однакові.")
    else:
        print("\nУВАГА: збурення змінює маску ReLU, різниця змішує дві гілки функції.")
    return all(diff <= TOL for *_, diff, _ in rows)


if __name__ == "__main__":
    import sys
    buggy = "--buggy" in sys.argv
    if buggy:
        print("Режим досліду: у градієнті за логітами прибрано ділення на N\n")
    report(divide_by_N=not buggy)

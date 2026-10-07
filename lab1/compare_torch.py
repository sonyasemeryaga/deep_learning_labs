import numpy as np
import torch
from torch import nn
from data import load_data
from network import init_params, forward, cross_entropy, backward

TOL = 1e-12

def torch_reference(X, y, W1, b1, W2, b2):
    lin1 = nn.Linear(4, 8, dtype=torch.float64)
    lin2 = nn.Linear(8, 3, dtype=torch.float64)
    with torch.no_grad():
        lin1.weight.copy_(torch.from_numpy(W1.T))
        lin1.bias.copy_(torch.from_numpy(b1))
        lin2.weight.copy_(torch.from_numpy(W2.T))
        lin2.bias.copy_(torch.from_numpy(b2))
    tX = torch.tensor(X, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.int64)
    Z2 = lin2(torch.relu(lin1(tX)))
    loss = nn.functional.cross_entropy(Z2, ty)
    loss.backward()
    grads = (lin1.weight.grad.numpy().T, lin1.bias.grad.numpy(),
             lin2.weight.grad.numpy().T, lin2.bias.grad.numpy())
    return float(loss.item()), grads

def compare(divide_by_N=True):
    X, y, _, _ = load_data()
    W1, b1, W2, b2 = init_params()
    Z2, cache = forward(X, W1, b1, W2, b2)
    loss_np = cross_entropy(Z2, y)
    grads_np = backward(cache, y, W2, divide_by_N=divide_by_N)
    loss_pt, grads_pt = torch_reference(X, y, W1, b1, W2, b2)
    rows = [("Втрата", abs(loss_np - loss_pt))]
    for name, g_np, g_pt in zip(["Градієнт W1", "Градієнт b1", "Градієнт W2", "Градієнт b2"],
                                grads_np, grads_pt):
        rows.append((name, np.abs(g_np - g_pt).max()))
    return loss_np, loss_pt, rows

def report(divide_by_N=True):
    loss_np, loss_pt, rows = compare(divide_by_N)
    print(f"Втрата NumPy:   {loss_np:.15f}")
    print(f"Втрата PyTorch: {loss_pt:.15f}")
    print(f"\n{'Величина':14} {'Макс. абс. різниця':>22}  Перевірку пройдено")
    for name, diff in rows:
        print(f"{name:14} {diff:>22.3e}  {'так' if diff <= TOL else 'НІ'}")
    passed = all(diff <= TOL for _, diff in rows)
    print(f"\nКритерій |a - b| <= {TOL:.0e}: "
          f"{'усі величини збігаються' if passed else 'є розбіжності'}")
    return passed


if __name__ == "__main__":
    import sys
    buggy = "--buggy" in sys.argv
    if buggy:
        print("Режим досліду: у градієнті за логітами прибрано ділення на N\n")
    report(divide_by_N=not buggy)

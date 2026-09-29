import numpy as np

N_IN, N_HIDDEN, N_OUT = 4, 8, 3
SEED = 0

def init_params():
    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, np.sqrt(2 / N_IN), size=(N_IN, N_HIDDEN))
    W2 = rng.normal(0.0, np.sqrt(2 / (N_HIDDEN + N_OUT)), size=(N_HIDDEN, N_OUT))
    b1 = np.zeros(N_HIDDEN)
    b2 = np.zeros(N_OUT)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return Z2, cache

def softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    E = np.exp(Z - m)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Z2, y):
    m = Z2.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(Z2 - m).sum(axis=1, keepdims=True))
    z_correct = Z2[np.arange(len(y)), y]
    return float(np.mean(-z_correct + lse.ravel()))

def loss_from_params(X, y, W1, b1, W2, b2):
    Z2, _ = forward(X, W1, b1, W2, b2)
    return cross_entropy(Z2, y)

def backward(cache, y, W2, divide_by_N=True):
    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    N = len(y)
    P = softmax(Z2)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0
    Delta2 = P - Y
    if divide_by_N:
        Delta2 = Delta2 / N
    dW2 = A1.T @ Delta2
    db2 = Delta2.sum(axis=0)
    dA1 = Delta2 @ W2.T
    Delta1 = dA1 * (Z1 > 0)
    dW1 = X.T @ Delta1
    db1 = Delta1.sum(axis=0)
    return dW1, db1, dW2, db2


if __name__ == "__main__":
    from data import load_data
    X, y, _, _ = load_data()
    W1, b1, W2, b2 = init_params()
    params = [("W1", W1, "He, sigma = sqrt(2/4)"),
              ("b1", b1, "нулі"),
              ("W2", W2, "Xavier, sigma = sqrt(2/(8+3))"),
              ("b2", b2, "нулі")]
    print("Параметри мережі Linear(4,8) -> ReLU -> Linear(8,3):")
    for name, value, scheme in params:
        print(f"  {name}  розмірність {str(value.shape):8}  {scheme}")
    print(f"  разом {sum(v.size for _, v, _ in params)} чисел")
    logits, cache = forward(X, W1, b1, W2, b2)
    probs = softmax(logits)
    loss = cross_entropy(logits, y)
    print("\nРозмірності прямого проходу:")
    for name in ["X", "Z1", "A1", "Z2"]:
        print(f"  {name:3} {cache[name].shape}")
    off = np.mean(cache["Z1"] <= 0)
    row_sum_error = np.abs(probs.sum(axis=1) - 1).max()
    accuracy = (logits.argmax(axis=1) == y).mean()
    print(f"\nВимкнених нейронів ReLU: {off:.1%} (очікувано близько половини)")
    print(f"Сума ймовірностей по рядку відрізняється від 1 не більше ніж на {row_sum_error:.1e}")
    print(f"\nВтрата на початкових вагах: {loss:.12f}")
    print(f"Частка правильних відповідей: {accuracy:.1%}")
    grads = list(zip(["dW1", "db1", "dW2", "db2"],
                     backward(cache, y, W2),
                     [W1, b1, W2, b2]))
    print("\nГрадієнти ручного зворотного проходу:")
    for name, grad, param in grads:
        same = "збігається з параметром" if grad.shape == param.shape else "РОЗБІЖНІСТЬ"
        print(f"  {name}  розмірність {str(grad.shape):8}  {same},"
              f"  норма {np.linalg.norm(grad):.6f}")

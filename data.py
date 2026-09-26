import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35
N_CLASSES = 3
SEED = 0

def load_data():
    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    rng = np.random.default_rng(SEED)
    train_idx = []
    test_idx = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(f"Iris: 150 об'єктів, {X_train.shape[1]} ознаки, {N_CLASSES} класи, обчислення у {X_train.dtype}\n")
    rows = [
        ("об'єктів усього", X_train.shape[0], X_test.shape[0]),
        ("об'єктів на клас", '/'.join(map(str, np.bincount(y_train))),
                            '/'.join(map(str, np.bincount(y_test)))),
    ]
    print(f"{'':18} {'train':>10} {'test':>10}")
    for label, a, b in rows:
        print(f"{label:18} {a:>10} {b:>10}")
    m_tr = X_train.mean(axis=0)
    s_tr = X_train.std(axis=0, ddof=0)
    m_te = X_test.mean(axis=0)
    s_te = X_test.std(axis=0, ddof=0)
    print("\nСтандартизація за статистиками train (ddof=0):")
    print(f"  train:  середнє за модулем не більше {np.abs(m_tr).max():.1e} (машинний нуль),"
          f" стандартне відхилення {s_tr.min():.3f}")
    print(f"  test:   середнє від {m_te.min():+.3f} до {m_te.max():+.3f},"
          f" стандартне відхилення від {s_te.min():.3f} до {s_te.max():.3f}")

import argparse
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from data import load_images, make_transform, part_rows, read_split
from export import load_model, make_session, onnx_logits

def predictions(path, x):
    return onnx_logits(make_session(path), x).argmax(1)

def paired(y, pred_fp32, pred):
    n = len(y)
    b = int(((pred == y) & (pred_fp32 != y)).sum())
    c = int(((pred != y) & (pred_fp32 == y)).sum())
    return {"b": b, "c": c, "delta_acc": (b - c) / n, "se_delta_acc": math.sqrt(b + c) / n,
            "beyond_noise": abs(b - c) > 2 * math.sqrt(b + c)}

def quality_row(y, pred_fp32, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)),
            "agreement_fp32": float((pred == pred_fp32).mean()),
            **paired(y, pred_fp32, pred)}

def single_se(y, pred):
    p = float(accuracy_score(y, pred))
    return math.sqrt(p * (1 - p) / len(y))

def quality_table(paths, x, y):
    preds = {name: predictions(path, x) for name, path in paths.items()}
    rows = {name: quality_row(y, preds["FP32"], pred) for name, pred in preds.items()}
    return rows, single_se(y, preds["FP32"])

def verdict(row):
    if row["b"] + row["c"] == 0:
        return "передбачення не змінились"
    if not row["beyond_noise"]:
        return "не відрізняється від шуму"
    return "краще за FP32 понад шум" if row["b"] > row["c"] else "гірше за FP32 понад шум"

def print_quality(rows, se_single):
    print(f"{'Варіант':16} {'macro-F1':>8} {'accuracy':>9} {'Згода з FP32':>13} {'b':>4} {'c':>4} "
          f"{'Δacc ± SE':>17}  Вердикт")
    for name, r in rows.items():
        delta = f"{r['delta_acc']:+.4f} ± {r['se_delta_acc']:.4f}"
        print(f"{name:16} {r['macro_f1']:>8.4f} {r['accuracy']:>9.4f} {r['agreement_fp32']:>13.4f} "
              f"{r['b']:>4} {r['c']:>4} {delta:>17}  {verdict(r) if name != 'FP32' else ''}")
    for name, r in rows.items():
        if name != "FP32" and r["se_delta_acc"] > 0:
            print(f"\n{name}: |b - c| = {abs(r['b'] - r['c'])}, 2√(b + c) = {2 * math.sqrt(r['b'] + r['c']):.2f}; "
                  f"SE однієї моделі √(p(1-p)/n) = {se_single:.4f}, "
                  f"у {se_single / r['se_delta_acc']:.1f} раза більша за парну")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--fp32", type=Path, default=Path("lab4_fp32.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("lab4_int8.onnx"))
    parser.add_argument("--int8-per-tensor", type=Path, default=Path("lab4_int8_per_tensor.onnx"))
    args = parser.parse_args()
    _, img_size = load_model(args.model)
    x, y = load_images(args.data, part_rows(read_split(args.split), "test"), "test", make_transform(img_size))
    paths = {"FP32": args.fp32, "INT8": args.int8}
    if args.int8_per_tensor.exists():
        paths["INT8 per-tensor"] = args.int8_per_tensor
    print_quality(*quality_table(paths, x, y))

import argparse
import math
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def conv_ranges(path):
    graph = onnx.load(str(path)).graph
    weights = {init.name: init for init in graph.initializer}
    rows = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        w = numpy_helper.to_array(weights[node.input[1]])
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        rows.append({"name": node.name, "shape": list(w.shape), "alpha_max": float(alpha.max()),
                     "alpha_median": float(np.median(alpha)), "r": r, "bits": math.log2(r)})
    return rows

def ranges_summary(rows):
    median_r = float(np.median([row["r"] for row in rows]))
    top = sorted(rows, key=lambda row: row["r"], reverse=True)[:3]
    return {"convs": rows, "top3": [{"name": row["name"], "r": row["r"]} for row in top],
            "median_r": median_r, "median_bits": math.log2(median_r)}

def print_ranges(summary):
    print(f"{'Згортка':48} {'Форма':>18} {'max α_c':>8} {'med α_c':>8} {'r':>6} {'log2 r':>7}")
    for row in summary["convs"]:
        shape = "×".join(map(str, row["shape"]))
        print(f"{row['name']:48} {shape:>18} {row['alpha_max']:>8.4f} {row['alpha_median']:>8.4f} "
              f"{row['r']:>6.2f} {row['bits']:>7.2f}")
    print("\nТри згортки з найбільшим r: " + ", ".join(f"{t['name']} ({t['r']:.2f})" for t in summary["top3"]))
    print(f"Медіана r за {len(summary['convs'])} згортками: {summary['median_r']:.2f}, "
          f"потензорний крок коштує медіанному каналу log2 r = {summary['median_bits']:.2f} біта")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path, default=Path("lab4_fp32.onnx"))
    args = parser.parse_args()
    print_ranges(ranges_summary(conv_ranges(args.fp32)))

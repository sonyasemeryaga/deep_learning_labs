import argparse
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from data import load_images, make_transform, part_rows, read_split
from export import load_model, make_session
from quantize import file_mb

ROUNDS = 3
WARMUP = 20
MEASURED = 100

def measure(session, feed, n):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        session.run(["logits"], feed)
        times.append(time.perf_counter() - start)
    return times

def protocol(sessions, x):
    feed = {"input": x}
    times = {name: [] for name in sessions}
    for _ in range(ROUNDS):
        for name, session in sessions.items():
            measure(session, feed, WARMUP)
            times[name].append(measure(session, feed, MEASURED))
    return {name: np.array(t) * 1000 for name, t in times.items()}

def summary(paths, times):
    rows = {}
    for name, t in times.items():
        rows[name] = {"file_mb": file_mb(paths[name]), "p50_ms": float(np.percentile(t, 50)),
                      "p95_ms": float(np.percentile(t, 95)),
                      "round_p50_ms": [float(np.percentile(r, 50)) for r in t], "times_ms": t.tolist()}
    fp32, int8 = rows["FP32"], rows["INT8"]
    return {"variants": rows,
            "size_ratio": fp32["file_mb"] / int8["file_mb"],
            "speedup_p50": fp32["p50_ms"] / int8["p50_ms"],
            "speedup_p95": fp32["p95_ms"] / int8["p95_ms"],
            "measured": max(int8["round_p50_ms"]) < min(fp32["round_p50_ms"])
                        or max(fp32["round_p50_ms"]) < min(int8["round_p50_ms"])}

def cpu_name():
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()

def environment():
    return {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "intra_op_threads": 1, "inter_op_threads": 1, "batch_size": 1}

def print_latency(s, env):
    print(f"{'Варіант':8} {'Файл, МБ':>9} {'p50, мс':>8} {'p95, мс':>8}  p50 за раундами, мс")
    for name, r in s["variants"].items():
        rounds = ", ".join(f"{v:.2f}" for v in r["round_p50_ms"])
        print(f"{name:8} {r['file_mb']:>9.2f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f}  {rounds}")
    print(f"\nРозмір файлу зменшився в {s['size_ratio']:.2f} раза, прискорення p50 {s['speedup_p50']:.2f}, "
          f"p95 {s['speedup_p95']:.2f}")
    print(f"Відрізки [min, max] p50 за раундами не перетинаються, зміна затримки виміряна: "
          f"{'так' if s['measured'] else 'ні'}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in env.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--fp32", type=Path, default=Path("lab4_fp32.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("lab4_int8.onnx"))
    args = parser.parse_args()
    _, img_size = load_model(args.model)
    x, _ = load_images(args.data, part_rows(read_split(args.split), "test")[:1], "test", make_transform(img_size))
    paths = {"FP32": args.fp32, "INT8": args.int8}
    sessions = {name: make_session(path) for name, path in paths.items()}
    print_latency(summary(paths, protocol(sessions, x)), environment())

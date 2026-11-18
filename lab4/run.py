import argparse
import json
from pathlib import Path
from data import calibration_rows, load_images, make_transform, part_rows, read_split
from data import report as data_report
from export import equivalence, export_onnx, load_model, make_session, print_equivalence
from latency import environment, print_latency, protocol, summary
from quality import print_quality, quality_table
from quantize import file_mb, prepare, quantize
from ranges import conv_ranges, print_ranges, ranges_summary

FP32 = Path("lab4_fp32.onnx")
PREP = Path("lab4_fp32_prep.onnx")
INT8 = Path("lab4_int8.onnx")
INT8_PER_TENSOR = Path("lab4_int8_per_tensor.onnx")

def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}\n")

def main(model_path, split_path, data_root, out):
    results = {}
    split = read_split(split_path)
    section("Контроль поділу й калібрувальної вибірки")
    results["checks"] = data_report(split)

    section("1. Експорт і перевірка відповідності")
    model, img_size = load_model(model_path)
    times = {"export_s": export_onnx(model, img_size, FP32)}
    transform = make_transform(img_size)
    x, y = load_images(data_root, part_rows(split, "test"), "test", transform)
    results["equivalence"] = equivalence(model, FP32, x, y)
    print_equivalence(results["equivalence"])

    section("2. Калібрування й квантизація")
    x_calibration, _ = load_images(data_root, calibration_rows(split), "train", transform)
    times["preprocess_s"] = prepare(FP32, PREP)
    times["quantize_s"] = quantize(PREP, INT8, x_calibration, per_channel=True)
    times["quantize_per_tensor_s"] = quantize(PREP, INT8_PER_TENSOR, x_calibration, per_channel=False)
    results["times"] = times
    results["file_mb"] = {"FP32": file_mb(FP32), "INT8": file_mb(INT8), "INT8 per-tensor": file_mb(INT8_PER_TENSOR)}
    print(f"Експорт: {times['export_s']:.2f} с, підготовка графа: {times['preprocess_s']:.2f} с, "
          f"квантизація: {times['quantize_s']:.2f} с (потензорна {times['quantize_per_tensor_s']:.2f} с)")
    print("Розмір файлів, МБ: " + ", ".join(f"{k} {v:.2f}" for k, v in results["file_mb"].items()))

    section("3–4. Якість і дослід із потензорними вагами")
    results["ranges"] = ranges_summary(conv_ranges(FP32))
    print_ranges(results["ranges"])
    print()
    rows, se_single = quality_table({"FP32": FP32, "INT8": INT8, "INT8 per-tensor": INT8_PER_TENSOR}, x, y)
    results["quality"] = {"variants": rows, "se_single_accuracy": se_single}
    print_quality(rows, se_single)

    section("5. Затримка")
    paths = {"FP32": FP32, "INT8": INT8}
    sessions = {name: make_session(path) for name, path in paths.items()}
    results["latency"] = summary(paths, protocol(sessions, x[:1]))
    results["environment"] = environment()
    print_latency(results["latency"], results["environment"])

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nУсі числа записано в {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--out", type=Path, default=Path("lab4_results.json"))
    args = parser.parse_args()
    main(args.model, args.split, args.data, args.out)

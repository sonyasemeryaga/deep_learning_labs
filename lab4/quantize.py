import argparse
import time
from collections import Counter
from pathlib import Path
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)
from data import calibration_rows, load_images, make_transform, read_split
from export import load_model

class Calibration(CalibrationDataReader):
    def __init__(self, x):
        self.x = x
        self.i = 0

    def get_next(self):
        if self.i == len(self.x):
            return None
        self.i += 1
        return {"input": self.x[self.i - 1:self.i]}

    def rewind(self):
        self.i = 0

def prepare(fp32, prep):
    start = time.perf_counter()
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    return time.perf_counter() - start

def quantize(prep, out, x_calibration, per_channel):
    start = time.perf_counter()
    quantize_static(str(prep), str(out), calibration_data_reader=Calibration(x_calibration),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel,
                    calibrate_method=CalibrationMethod.MinMax)
    return time.perf_counter() - start

def file_mb(path):
    return Path(path).stat().st_size / 1e6

def qdq_counts(path):
    ops = Counter(node.op_type for node in onnx.load(str(path)).graph.node)
    return {op: ops[op] for op in ["QuantizeLinear", "DequantizeLinear", "Conv"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--fp32", type=Path, default=Path("lab4_fp32.onnx"))
    parser.add_argument("--prep", type=Path, default=Path("lab4_fp32_prep.onnx"))
    parser.add_argument("--out", type=Path, default=Path("lab4_int8.onnx"))
    parser.add_argument("--per-tensor", action="store_true", help="потензорні ваги замість поканальних")
    args = parser.parse_args()
    _, img_size = load_model(args.model)
    x_calibration, _ = load_images(args.data, calibration_rows(read_split(args.split)), "train",
                                   make_transform(img_size))
    print(f"Калібрувальна вибірка: {x_calibration.shape}")
    t_prep = prepare(args.fp32, args.prep)
    t_quant = quantize(args.prep, args.out, x_calibration, per_channel=not args.per_tensor)
    print(f"Підготовка графа: {t_prep:.2f} с, калібрування й квантизація: {t_quant:.2f} с")
    print(f"Розмір: {args.fp32} {file_mb(args.fp32):.2f} МБ, {args.out} {file_mb(args.out):.2f} МБ")
    print(f"Вузли в {args.out}: {qdq_counts(args.out)}")

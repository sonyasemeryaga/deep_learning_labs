import argparse
import time
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import resnet18
from data import CLASSES, load_images, make_transform, part_rows, read_split

BATCH = 64
OPSET = 17
MAX_LOGIT_DIFF = 1e-3

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError(f"Порядок класів у моделі {checkpoint['classes']} не збігається з {CLASSES}")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["img_size"]

def export_onnx(model, img_size, path):
    start = time.perf_counter()
    torch.onnx.export(model, torch.zeros(1, 3, img_size, img_size), path, dynamo=False, opset_version=OPSET,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    return time.perf_counter() - start

def torch_logits(model, x):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + BATCH])).numpy() for i in range(0, len(x), BATCH)])

def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

def onnx_logits(session, x):
    return np.concatenate([session.run(["logits"], {"input": x[i:i + BATCH]})[0] for i in range(0, len(x), BATCH)])

def op_counts(path):
    return Counter(node.op_type for node in onnx.load(str(path)).graph.node)

def macro_f1(y, logits):
    return float(f1_score(y, logits.argmax(1), average="macro"))

def equivalence(model, path, x, y):
    z_torch = torch_logits(model, x)
    z_onnx = onnx_logits(make_session(path), x)
    ops = op_counts(path)
    return {
        "max_abs_logit_diff": float(np.abs(z_onnx - z_torch).max()),
        "argmax_agreement": float((z_onnx.argmax(1) == z_torch.argmax(1)).mean()),
        "macro_f1_pytorch": macro_f1(y, z_torch),
        "macro_f1_onnx_fp32": macro_f1(y, z_onnx),
        "conv_nodes": ops["Conv"],
        "batchnorm_nodes": ops["BatchNormalization"],
    }

def print_equivalence(r):
    ok = (r["max_abs_logit_diff"] < MAX_LOGIT_DIFF and r["argmax_agreement"] == 1.0
          and abs(r["macro_f1_pytorch"] - r["macro_f1_onnx_fp32"]) < 1e-12
          and r["conv_nodes"] == 20 and r["batchnorm_nodes"] == 0)
    print(f"Максимальна різниця логітів |z_ONNX - z_PyTorch|: {r['max_abs_logit_diff']:.2e}")
    print(f"Частка зображень зі збігом класу: {r['argmax_agreement']:.4f}")
    print(f"macro-F1 на test: PyTorch {r['macro_f1_pytorch']:.4f}, ONNX FP32 {r['macro_f1_onnx_fp32']:.4f}")
    print(f"Вузлів Conv: {r['conv_nodes']}, вузлів BatchNormalization: {r['batchnorm_nodes']}")
    print(f"\nФайл ONNX обчислює ту саму функцію, що й модель PyTorch: {'так' if ok else 'НІ'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="lab3_model.pt")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Тека з seg_train/ і seg_test/")
    parser.add_argument("--out", type=Path, default=Path("lab4_fp32.onnx"))
    args = parser.parse_args()
    model, img_size = load_model(args.model)
    seconds = export_onnx(model, img_size, args.out)
    print(f"Модель експортовано в {args.out} за {seconds:.2f} с\n")
    x, y = load_images(args.data, part_rows(read_split(args.split), "test"), "test", make_transform(img_size))
    print_equivalence(equivalence(model, args.out, x, y))

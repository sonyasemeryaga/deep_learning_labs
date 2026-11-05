import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

REGIMES = ["scratch", "frozen", "unfreeze"]
NUM_CLASSES = 6
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY = 1e-4
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
FROZEN = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
EXPECTED_TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}
EXPECTED_TOTAL = 11179590

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_model(regime, seed, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, NUM_CLASSES)
    for name in FROZEN[regime]:
        getattr(model, name).requires_grad_(False)
    model.to(device)
    return model, make_optimizer(model, regime)

def make_optimizer(model, regime):
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": LR_HEAD}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD},
                  {"params": list(model.layer4.parameters()), "lr": LR_BACKBONE}]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

def set_train_mode(model, regime):
    model.train()
    for name in FROZEN[regime]:
        getattr(model, name).eval()

def count(params):
    return sum(p.numel() for p in params)

def parameter_summary(model, optimizer):
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "total": count(model.parameters()),
        "trainable": count(p for p in model.parameters() if p.requires_grad),
        "groups": [(count(g["params"]), g["lr"]) for g in optimizer.param_groups],
        "frozen_in_optimizer": len(in_optimizer - trainable),
        "trainable_outside_optimizer": len(trainable - in_optimizer),
    }


def report(seed=1):
    print(f"{'Режим':10} {'Усього':>11} {'Навчається':>11}  {'Групи AdamW (параметрів, lr)':34} {'Збіг':>5}")
    ok = True
    for regime in REGIMES:
        model, optimizer = build_model(regime, seed)
        s = parameter_summary(model, optimizer)
        groups = ", ".join(f"{n} при {lr:g}" for n, lr in s["groups"])
        match = (s["total"] == EXPECTED_TOTAL and s["trainable"] == EXPECTED_TRAINABLE[regime]
                 and s["frozen_in_optimizer"] == 0 and s["trainable_outside_optimizer"] == 0)
        ok &= match
        print(f"{regime:10} {s['total']:>11} {s['trainable']:>11}  {groups:34} {'так' if match else 'ні':>5}")
    print("\nЧисла збігаються з таблицею режимів, а в оптимізатор потрапили рівно параметри "
          f"з requires_grad=True: {'так' if ok else 'НІ'}")
    return ok


if __name__ == "__main__":
    report()

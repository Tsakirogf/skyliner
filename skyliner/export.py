"""Export trained .pt weights to ONNX (runtime needs only onnxruntime + numpy + opencv).

  python -m skyliner.export --runs runs --fold all --out models
"""
import argparse
import hashlib
import json
import os

import torch

from .models import SCENE_H, SCENE_W, EdgeNet, SceneNet


def export(runs, fold, out):
    os.makedirs(out, exist_ok=True)
    info = {}
    for name, cls, shape, dyn in [
        ("scene", SceneNet, (1, 3, SCENE_H, SCENE_W), None),
        ("edge", EdgeNet, (1, 3, 256, 512), {"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}),
    ]:
        net = cls()
        net.load_state_dict(torch.load(os.path.join(runs, f"{name}_{fold}.pt"), map_location="cpu"))
        net.eval()
        path = os.path.join(out, f"{name}_{fold}.onnx" if fold != "all" else f"{name}.onnx")
        torch.onnx.export(net, torch.zeros(shape), path, input_names=["x"], output_names=["y"],
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:10]
        info[name] = {"file": os.path.basename(path), "sha256_10": h}
        print("wrote", path, h)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    info = export(a.runs, a.fold, a.out)
    if a.fold == "all":
        with open(os.path.join(a.out, "models.json"), "w") as f:
            json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()

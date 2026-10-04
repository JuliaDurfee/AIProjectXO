"""
Check all three classifiers the way the playground will run them, then export .onnx files.

    python export_playground.py

1. Loads test_data/O and test_data/X through the playground's exact input format
   (whole image -> 64x64 gray in [-1,1]) and prints each model's accuracy.
2. Exports onnx/perceptron.onnx, onnx/mlp.onnx, onnx/cnn.onnx with the playground's helper,
   which checks the ONNX file and compares it with PyTorch.
Copy the .onnx files to your phone and choose them in the playground.
"""
import glob
import os

import torch

from models import build
from playground import PerceptronNet, PlaygroundModel, contract_tensor
from python.playground_export import export_for_playground  # from the model-playground submodule

CLASSES = ["O", "X"]


def load_test(folder="test_data"):
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        for p in sorted(glob.glob(os.path.join(folder, name, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                xs.append(contract_tensor(p))
                ys.append(label)
    return torch.stack(xs), torch.tensor(ys)


def classifier(name):
    if name == "perceptron":
        return PerceptronNet()
    m = build(name)
    m.load_state_dict(torch.load(f"models/{name}.pt", map_location="cpu", weights_only=True))
    return m


def main():
    X, y = load_test()
    print(f"test images: {len(y)} ({(y == 0).sum().item()} O, {(y == 1).sum().item()} X)")
    os.makedirs("onnx", exist_ok=True)
    samples = [X[i:i + 1] for i in range(0, len(X), max(1, len(X) // 8))]

    for name in ["perceptron", "mlp", "cnn"]:
        model = PlaygroundModel(classifier(name)).eval()
        with torch.no_grad():
            pred = model(X).argmax(1)
        print(f"{name:10s} accuracy on test_data (playground input): {(pred == y).float().mean():.3f}")
        export_for_playground(model, f"onnx/{name}.onnx", sample_inputs=samples)


if __name__ == "__main__":
    main()
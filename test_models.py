import os
import glob
import numpy as np
import torch

from preprocess import preprocess
from perceptron import predict as perceptron_predict
from models import build

CLASSES = ["O", "X"]
TEST_DIR = "test_data"


def load_test_data():
    X = []
    y = []
    paths = []

    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(TEST_DIR, class_name)

        for path in sorted(glob.glob(os.path.join(folder, "*"))):
            img = preprocess(path)

            if img is None:
                print("Skipped:", path)
                continue

            X.append(img)
            y.append(label)
            paths.append(path)

    if len(X) == 0:
        raise RuntimeError("No test images found.")

    return (
        np.stack(X).astype(np.float32),
        np.array(y, dtype=np.int64),
        paths
    )


def test_perceptron(X, y, paths):
    correct = 0
    mistakes = []

    for img, label, path in zip(X, y, paths):
        prediction_name = perceptron_predict(img)

        prediction = 1 if prediction_name == "X" else 0

        if prediction == label:
            correct += 1
        else:
            mistakes.append(
                (path, CLASSES[label], CLASSES[prediction])
            )

    accuracy = correct / len(y)

    return accuracy, mistakes


def test_neural_model(model_name, X, y, paths, device):
    model = build(model_name).to(device)

    model.load_state_dict(
        torch.load(
            f"models/{model_name}_final.pt",
            map_location=device
        )
    )

    model.eval()

    correct = 0
    mistakes = []

    with torch.no_grad():
        for img, label, path in zip(X, y, paths):
            x = torch.from_numpy(img).float()

            # [64, 64] -> [1, 1, 64, 64]
            x = x.unsqueeze(0).unsqueeze(0).to(device)

            outputs = model(x)

            prediction = outputs.argmax(dim=1).item()

            if prediction == label:
                correct += 1
            else:
                mistakes.append(
                    (path, CLASSES[label], CLASSES[prediction])
                )

    accuracy = correct / len(y)

    return accuracy, mistakes


def print_results(name, accuracy, mistakes):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Mistakes: {len(mistakes)}")

    if mistakes:
        print("\nMisclassified images:")

        for path, actual, predicted in mistakes:
            print(
                f"{path} | actual={actual} | predicted={predicted}"
            )


def main():
    X, y, paths = load_test_data()

    print(f"Total test images: {len(y)}")
    print(f"O images: {(y == 0).sum()}")
    print(f"X images: {(y == 1).sum()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # Perceptron
    accuracy, mistakes = test_perceptron(
        X,
        y,
        paths
    )

    print_results(
        "MANUAL PERCEPTRON",
        accuracy,
        mistakes
    )

    # MLP
    accuracy, mistakes = test_neural_model(
        "mlp",
        X,
        y,
        paths,
        device
    )

    print_results(
        "MLP",
        accuracy,
        mistakes
    )

    # CNN
    accuracy, mistakes = test_neural_model(
        "cnn",
        X,
        y,
        paths,
        device
    )

    print_results(
        "CNN",
        accuracy,
        mistakes
    )


if __name__ == "__main__":
    main()
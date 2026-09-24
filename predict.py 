"""
Run all three classifiers on new photos (use this in class).
 
    python predict.py photo.jpg [more.jpg ...]
"""
import sys
 
import torch
 
import perceptron
from dataset import CLASSES
from models import build
from preprocess import preprocess
 
 
def load(name):
    m = build(name)
    m.load_state_dict(torch.load(f"models/{name}.pt", map_location="cpu"))
    return m.eval()
 
 
def main(paths):
    mlp, cnn = load("mlp"), load("cnn")
    for p in paths:
        x = preprocess(p)
        if x is None:
            print(f"{p}: no ink found"); continue
        t = torch.from_numpy(x)[None, None]          # (1, 1, SIZE, SIZE)
        with torch.no_grad():
            pm = torch.softmax(mlp(t), 1)[0]
            pc = torch.softmax(cnn(t), 1)[0]
        print(f"{p}:  perceptron={perceptron.predict(x)} (score {perceptron.score(x):+.1f})"
              f"  MLP={CLASSES[pm.argmax()]} ({pm.max():.2f})"
              f"  CNN={CLASSES[pc.argmax()]} ({pc.max():.2f})")
 
 
if __name__ == "__main__":
    main(sys.argv[1:])

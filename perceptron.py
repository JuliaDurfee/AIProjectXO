"""
Classifier 1: a single perceptron with HAND-PICKED weights (no training).
 
    score = sum(W * x) + b        predict X if score > 0, else O
 
x is the SIZE x SIZE ink map from preprocess.py (the shape is cropped to its bounding box,
so it roughly fills the image). Coordinates u, v run from -1 to +1 across the image.
 
Reasoning behind the weights (this is "what the perceptron needs to learn"):
  +1  CENTRE disk          an X's two strokes cross here; an O is hollow
  +1  the four CORNERS     an X's arms end in the corners of its bounding box;
                           an O's curve passes the diagonals closer in (~0.59), outside these
  -1  the four EDGE MIDDLES an O passes through top/bottom/left/right middle; an X doesn't
Rough ink lengths: X gets ~ +2.2 and O ~ -2.4 (in stroke-length units), so bias b = 0
sits naturally in the middle. You may nudge b by hand after looking at the scores.
 
Run:  python perceptron.py            -> accuracy on your dataset + weights.png
"""
import numpy as np
 
from preprocess import SIZE
 
BIAS = 0.0
 
 
def make_weights(size=SIZE):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    u, v = (xx - c) / c, (yy - c) / c
    r = np.sqrt(u ** 2 + v ** 2)
    W = np.zeros((size, size), np.float32)
    W[r < 0.30] += 1.0                                      # centre
    W[(np.abs(u) > 0.65) & (np.abs(v) > 0.65)] += 1.0       # corners
    W[(np.abs(u) < 0.30) & (np.abs(v) > 0.65)] -= 1.0       # top / bottom middle
    W[(np.abs(v) < 0.30) & (np.abs(u) > 0.65)] -= 1.0       # left / right middle
    return W
 
 
W = make_weights()
 
 
def score(x):
    return float((W * x).sum() + BIAS)
 
 
def predict(x):
    return "X" if score(x) > 0 else "O"
 
 
if __name__ == "__main__":
    import cv2
    from dataset import CLASSES, load_cached
 
    cv2.imwrite("weights.png", cv2.resize(((W + 1) * 127.5).astype(np.uint8), (280, 280),
                                          interpolation=cv2.INTER_NEAREST))
    X, y, _ = load_cached()
    s = np.array([score(x) for x in X])
    pred = (s > 0).astype(int)
    print(f"accuracy on dataset: {(pred == y).mean():.3f}  (n={len(y)})")
    for k, name in enumerate(CLASSES):
        print(f"  {name}: mean score {s[y == k].mean():+.1f}, min {s[y == k].min():+.1f}, max {s[y == k].max():+.1f}")
    print("Saved weights.png (white = +1, grey = 0, black = -1)")

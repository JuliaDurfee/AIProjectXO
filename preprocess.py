"""
Turns a raw whiteboard photo into a small SIZE x SIZE "ink map" (float32, 0 = board, 1 = ink).
 
This is where the marker-colour and lighting twists are handled, so every classifier
sees normalised input:
  1. Estimate the whiteboard background per colour channel with a morphological CLOSING
     (removes thin dark strokes, keeps slow lighting changes such as shadows / glare).
  2. Ink = how much darker than the local background a pixel is, max over R,G,B,
     divided by local brightness  -> works for any marker colour and any lighting.
  3. Otsu threshold (picks the cut-off per image) + remove small specks.
  4. Crop to the bounding box of the big ink blobs, resize to SIZE x SIZE.
 
Debug:  python preprocess.py photo1.jpg photo2.jpg ...   -> writes debug/<name>_steps.png
"""
import os
import sys
 
import cv2
import numpy as np
 
SIZE = 64          # final image side; try 16, 20, 28, 32
WORK_SIDE = 800    # photos are shrunk to this longest side first (speed + consistent kernels)
BG_KERNEL = 41     # must be clearly wider than a marker stroke at WORK_SIDE resolution
PAD = 0.12         # margin around the shape's bounding box (fraction of box size)
 
 
def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
 
 
def shrink(bgr):
    h, w = bgr.shape[:2]
    s = WORK_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr
 
 
def ink_map(bgr):
    """Float map, large where there is marker ink, ~0 on the board. Colour/lighting invariant."""
    img = bgr.astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL, BG_KERNEL))
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)       # board without strokes
    diff = np.clip(bg - img, 0, None).max(axis=2)        # darker than board in ANY channel
    return diff / (bg.mean(axis=2) + 10.0)               # relative -> brightness invariant
 
 
def binarize(ink):
    ink = cv2.GaussianBlur(ink, (5, 5), 0)
    ink8 = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(ink8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks
    return bw
 
 
def crop_shape(bw, min_frac=0.15):
    """Keep connected blobs at least min_frac of the biggest one (an X may be 2 blobs),
    crop their joint bounding box (+PAD) and return it. None if no ink found."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= min_frac * areas.max())
    mask = np.isin(labels, keep).astype(np.uint8) * 255
 
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0 + 1) * PAD) + 1, int((x1 - x0 + 1) * PAD) + 1
    m = max(py, px)
    mask = cv2.copyMakeBorder(mask, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    return mask[y0 + m - py: y1 + m + py + 1, x0 + m - px: x1 + m + px + 1]
 
 
def preprocess(src, return_steps=False):
    """src: file path or BGR array. Returns SIZE x SIZE float32 in [0,1], or None."""
    bgr = load_bgr(src) if isinstance(src, str) else src
    bgr = shrink(bgr)
    ink = ink_map(bgr)
    bw = binarize(ink)
    crop = crop_shape(bw)
    if crop is None:
        out = None
    else:
        # Thicken a little so thin strokes survive downscaling, then stretch to a square.
        crop = cv2.dilate(crop, np.ones((5, 5), np.uint8))
        out = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if return_steps:
        return out, (bgr, ink, bw)
    return out
 
 
def _debug(paths):
    os.makedirs("debug", exist_ok=True)
    for p in paths:
        out, (bgr, ink, bw) = preprocess(p, return_steps=True)
        h = 300
        def fit(im):
            im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if im.dtype != np.uint8 else im
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        final = np.zeros((SIZE, SIZE), np.float32) if out is None else out
        panel = np.hstack([fit(bgr), fit(ink), fit(bw), fit((final * 255).astype(np.uint8))])
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join("debug", f"{name}_steps.png"), panel)
        print(p, "->", "NO INK FOUND" if out is None else "ok")
 
 
if __name__ == "__main__":
    _debug(sys.argv[1:])

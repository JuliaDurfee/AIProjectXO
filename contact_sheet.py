"""
Contact sheets: every image in a folder on a few grid pages, so you can review
the whole dataset at a glance.

Each tile shows the photo (left), the 64x64 ink map the models see (right),
the file name and the perceptron score. Red border = perceptron gets it wrong,
orange = preprocessing found no ink.

    python contact_sheet.py data/O data/X                # all images
    python contact_sheet.py data/O data/X --sort score   # most borderline first
    python contact_sheet.py data/O data/X --wrong        # only perceptron mistakes

Pages are saved to review_sheets/<folder>_page01.png, ...
The folder name decides the expected label: a folder named O or X.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import perceptron
from preprocess import preprocess

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
THUMB, INK, COLS, ROWS = 140, 70, 8, 6
TILE_W, TILE_H = THUMB + INK + 34, THUMB + 34


def fit(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 255, np.uint8)
    y, x = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def make_tile(path, expected):
    bgr = cv2.imread(str(path))
    tile = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    if bgr is None:
        return tile, None, True
    x = preprocess(bgr)
    tile[4:4 + THUMB, 4:4 + THUMB] = fit(bgr, THUMB)
    if x is None:
        score, color, bad = None, (0, 140, 255), True               # orange: no ink
    else:
        ink = cv2.resize((255 - x * 255).astype(np.uint8), (INK, INK), interpolation=cv2.INTER_NEAREST)
        tile[4:4 + INK, THUMB + 8:THUMB + 8 + INK] = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
        score = perceptron.score(x)
        bad = (score > 0) != (expected == "X")
        color = (0, 0, 230) if bad else (200, 200, 200)             # red: perceptron wrong
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), color, 3 if bad else 1)
    label = path.name if len(path.name) <= 26 else path.name[:23] + "..."
    cv2.putText(tile, label, (6, THUMB + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    text = "no ink" if score is None else f"score {score:+.1f}"
    cv2.putText(tile, text, (THUMB + 8, INK + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color if bad else (80, 80, 80), 1, cv2.LINE_AA)
    return tile, score, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--sort", choices=["name", "score"], default="name")
    ap.add_argument("--wrong", action="store_true", help="only show perceptron mistakes / no-ink images")
    ap.add_argument("--out", default="review_sheets")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)

    for folder in map(Path, args.folders):
        expected = folder.name.upper()
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        tiles = [(p,) + make_tile(p, expected) for p in files]
        if args.wrong:
            tiles = [t for t in tiles if t[3]]
        if args.sort == "score":   # closest to the decision boundary (score 0) first
            tiles.sort(key=lambda t: -1 if t[2] is None else abs(t[2]))
        per_page = COLS * ROWS
        pages = max(1, -(-len(tiles) // per_page))
        for page in range(pages):
            chunk = tiles[page * per_page:(page + 1) * per_page]
            sheet = np.full((ROWS * TILE_H, COLS * TILE_W, 3), 255, np.uint8)
            for i, (_, tile, _, _) in enumerate(chunk):
                r, c = divmod(i, COLS)
                sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
            used_rows = max(1, -(-len(chunk) // COLS))
            out = Path(args.out) / f"{folder.name}_page{page + 1:02d}.png"
            cv2.imwrite(str(out), sheet[:used_rows * TILE_H])
        wrong = sum(t[3] for t in tiles)
        print(f"{folder}: {len(tiles)} images on {pages} page(s), {wrong} flagged -> {args.out}/")


if __name__ == "__main__":
    main()
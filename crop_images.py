from pathlib import Path
import cv2
import numpy as np
import csv

RAW_DIR = Path("raw_images")
OUT_DIR = Path("cropped_unlabeled")
OUT_DIR.mkdir(exist_ok=True)

def find_candidates(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 20).astype(np.uint8) * 255

    mask[: int(0.035 * H), :] = 0
    mask[int(0.94 * H):, :] = 0
    mask[:, : int(0.02 * W)] = 0
    mask[:, int(0.985 * W):] = 0

    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if not (16 <= w <= 180 and 16 <= h <= 180):
            continue
        if not (120 <= area <= 6000 and w * h >= 300):
            continue
        if max(w / h, h / w) > 3.8:
            continue
        boxes.append((x, y, w, h))
    return boxes

rows = []
counter = 0

for source_num, path in enumerate(sorted(RAW_DIR.glob("*")), start=1):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    img = cv2.imread(str(path))
    if img is None:
        continue

    H, W = img.shape[:2]
    for x, y, w, h in find_candidates(img):
        pad = max(10, int(0.28 * max(w, h)))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(W, x+w+pad), min(H, y+h+pad)
        crop = img[y0:y1, x0:x1]

        counter += 1
        name = f"crop_{counter:04d}_src{source_num:02d}.jpg"
        cv2.imwrite(str(OUT_DIR / name), crop)
        rows.append([name, path.name, source_num, x0, y0, x1-x0, y1-y0])

with open("manifest.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["crop_file", "source_image", "source_number", "x", "y", "width", "height"])
    writer.writerows(rows)

print(f"Saved {counter} candidate crops to {OUT_DIR}/")

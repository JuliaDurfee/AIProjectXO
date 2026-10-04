"""
Stress test: how do the three classifiers cope with conditions our dataset doesn't contain?

    python stress_test.py              # test_data only (97 photos, never trained on)
    python stress_test.py --all        # also include the training photos (data/)

Each photo is altered in a way class conditions may differ (lighting, marker colour,
blur, glare, loose crops, full-resolution thin strokes, ...), then goes through EXACTLY
the playground path: whole photo -> 64x64 gray (python.common.preprocess) -> model.
For every condition it prints, per model:
    acc   = fraction classified correctly
    conf  = average probability given to the CORRECT class (low = unsure, even if right)
It also saves stress_examples.png showing what each condition looks like.

No training happens here; the models are only evaluated.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["O", "X"]
SEED = 0


# ---------------------------------------------------------------- photo alterations
# Each takes a PIL RGB image and a numpy random generator, returns a PIL RGB image.

def _arr(im):
    return np.asarray(im).astype(np.float32)


def _img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _board(im):
    """Board colour = per-channel median (most of each crop is board)."""
    return np.median(_arr(im).reshape(-1, 3), axis=0)


def brightness(f):
    return lambda im, rng: _img(_arr(im) * f)


def uneven_light(lo):
    """Light falls off across the photo from 1.0 to `lo`, in a random direction."""
    def f(im, rng):
        a = _arr(im)
        h, w = a.shape[:2]
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        t = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        return _img(a * (lo + (1 - lo) * t)[..., None])
    return f


def faded_ink(keep):
    """Ink only `keep` as dark as before (light marker, e.g. yellow/orange, or drying out)."""
    def f(im, rng):
        a, b = _arr(im), _board(im)
        return _img(b + (a - b) * keep)
    return f


def swap_colour(im, rng):
    """Different marker colour: permute RGB channels (red -> blue/green etc.)."""
    a = _arr(im)
    return _img(a[..., [2, 0, 1]])


def blur(frac):
    return lambda im, rng: im.filter(ImageFilter.GaussianBlur(radius=frac * max(im.size)))


def noise(sigma):
    return lambda im, rng: _img(_arr(im) + rng.normal(0, sigma, _arr(im).shape))


def glare(im, rng):
    """Bright ceiling-light reflection near (not on) the shape."""
    a = _arr(im)
    h, w = a.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    cy, cx = h / 2 + 0.38 * h * np.sin(ang), w / 2 + 0.38 * w * np.cos(ang)
    yy, xx = np.mgrid[0:h, 0:w]
    r = 0.18 * max(h, w)
    mask = np.exp(-(((yy - cy) / r) ** 2 + ((xx - cx) / (1.6 * r)) ** 2))[..., None]
    return _img(a + (255 - a) * 0.9 * mask)


def loose_crop(fill, offcentre=False):
    """Shape fills only `fill` of the frame width (photo not cropped tightly)."""
    def f(im, rng):
        w, h = im.size
        W, H = int(w / fill), int(h / fill)
        b = _board(im)
        canvas = np.ones((H, W, 3), np.float32) * b + rng.normal(0, 3, (H, W, 3))
        if offcentre:
            x0 = int(rng.uniform(0.05, 0.95) * (W - w))
            y0 = int(rng.uniform(0.05, 0.95) * (H - h))
        else:
            x0, y0 = (W - w) // 2, (H - h) // 2
        canvas[y0:y0 + h, x0:x0 + w] = _arr(im)
        return _img(canvas)
    return f


def portrait(im, rng):
    """Uncropped 3:4 phone photo: the playground squashes it to a square."""
    w, h = im.size
    H = int(w * 4 / 3) if w >= h else h
    W = max(w, int(H * 3 / 4))
    b = _board(im)
    canvas = np.ones((H, W, 3), np.float32) * b
    canvas[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = _arr(im)
    return _img(canvas)


def full_res_thin(im, rng, scale=10, thin=15):
    """Full-resolution phone photo: big image, sharp THIN strokes. The browser's 64x64
    sampling of such photos breaks strokes into dots (the failure we saw in class tests).
    Upscale smoothly, then shrink the dark strokes with a max filter."""
    w, h = im.size
    big = _arr(im.resize((w * scale, h * scale), Image.BICUBIC))
    return _img(_max_filter(big, thin))


def _max_filter(a, k):
    """Fast separable k x k max filter (PIL's MaxFilter is far too slow on big images)."""
    r = k // 2
    for axis in (0, 1):
        p = np.pad(a, [(r, r) if i == axis else (0, 0) for i in range(3)], mode="edge")
        n = a.shape[axis]
        out = np.take(p, range(0, n), axis=axis)
        for d in range(1, k):
            out = np.maximum(out, np.take(p, range(d, d + n), axis=axis))
        a = out
    return a


def classroom(im, rng):
    """Several at once: loose crop + full-res thin strokes + uneven light + glare."""
    im = loose_crop(0.5)(im, rng)
    im = full_res_thin(im, rng, scale=6, thin=9)
    im = uneven_light(0.55)(im, rng)
    return glare(im, rng)


CONDITIONS = [
    ("baseline (unchanged)", lambda im, rng: im),
    ("darker x0.5", brightness(0.5)),
    ("brighter x1.4", brightness(1.4)),
    ("uneven light 1.0->0.4", uneven_light(0.4)),
    ("faded ink 50%", faded_ink(0.5)),
    ("faded ink 30%", faded_ink(0.3)),
    ("other marker colour", swap_colour),
    ("blur small", blur(0.01)),
    ("blur strong", blur(0.025)),
    ("noise sigma 12", noise(12)),
    ("glare spot", glare),
    ("loose crop: fills 60%", loose_crop(0.6)),
    ("loose crop: fills 40%", loose_crop(0.4)),
    ("loose + off-centre 50%", loose_crop(0.5, offcentre=True)),
    ("uncropped 3:4 photo", portrait),
    ("full-res thin strokes", full_res_thin),
    ("CLASSROOM combo", classroom),
]


# ---------------------------------------------------------------- data + evaluation

def list_images(folders):
    items = []
    for folder in folders:
        for label, name in enumerate(CLASSES):
            for p in sorted(glob.glob(os.path.join(folder, name, "*"))):
                if p.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((p, label))
    return items


def example_sheet(items, path="stress_examples.png", thumb=96):
    """One O and one X under every condition, so you can see what was simulated."""
    picks = [next(p for p, l in items if l == 0), next(p for p, l in items if l == 1)]
    pad, label_h = 6, 28
    W = len(CONDITIONS) * (thumb + pad) + pad
    H = len(picks) * (thumb + pad) + label_h + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    for ci, (cname, fn) in enumerate(CONDITIONS):
        x = pad + ci * (thumb + pad)
        draw.text((x, 2), cname[:16], fill="black")
        draw.text((x, 14), cname[16:32], fill="black")
        for ri, p in enumerate(picks):
            im = fn(Image.open(p).convert("RGB"), np.random.default_rng(SEED + ri))
            sheet.paste(im.resize((thumb, thumb), Image.BILINEAR),
                        (x, label_h + ri * (thumb + pad)))
    sheet.save(path)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also use training photos in data/")
    args = ap.parse_args()

    import torch
    from playground import PlaygroundModel, contract_preprocess
    from export_playground import classifier

    items = list_images(["test_data"] + (["data"] if args.all else []))
    y = torch.tensor([l for _, l in items])
    photos = [Image.open(p).convert("RGB") for p, _ in items]
    print(f"{len(items)} photos ({(y == 0).sum().item()} O, {(y == 1).sum().item()} X)\n")

    names = ["perceptron", "mlp", "cnn"]
    models = {n: PlaygroundModel(classifier(n)).eval() for n in names}

    header = f"{'condition':26s}" + "".join(f"{n:>18s}" for n in names)
    print(header)
    print(f"{'':26s}" + "".join(f"{'acc   conf':>18s}" for _ in names))
    print("-" * len(header))
    for cname, fn in CONDITIONS:
        rng = np.random.default_rng(SEED)
        X = torch.stack([contract_preprocess(fn(im, rng)) for im in photos])
        row = f"{cname:26s}"
        with torch.no_grad():
            for n in names:
                prob = torch.softmax(models[n](X), dim=1)
                acc = (prob.argmax(1) == y).float().mean().item()
                conf = prob[torch.arange(len(y)), y].mean().item()
                row += f"{acc:>11.3f} {conf:5.2f} "
        print(row)

    example_sheet(items)


if __name__ == "__main__":
    main()
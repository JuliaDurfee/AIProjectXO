"""
Photo-level alterations that imitate classroom conditions our dataset doesn't contain:
lighting, marker colour/fading, blur, noise, glare, loose crops, full-resolution thin strokes.

Used in two places:
  * stress_test.py  - FIXED settings, to measure how robust the models are
  * dataset.py      - random_photo_aug() with RANDOM settings, to create extra training
                      versions of every photo BEFORE ink extraction (photo-level augmentation)

Every function takes a PIL RGB image and a numpy random generator and returns a PIL RGB image.
"""
import numpy as np
from PIL import Image, ImageFilter


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


# ---------------------------------------------------------------- random training augmentation

def _odd(v):
    v = max(3, int(round(v)))
    return v if v % 2 else v + 1


def random_photo_aug(im, rng):
    """One random 'classroom-like' version of a photo for TRAINING.
    Each effect is applied with some probability and a random strength, so the
    combinations vary. Settings are random ranges, not the stress test's fixed values."""
    if rng.random() < 0.3:
        im = swap_colour_random(im, rng)
    if rng.random() < 0.4:
        im = faded_ink(rng.uniform(0.45, 0.9))(im, rng)
    if rng.random() < 0.5:
        im = loose_crop(rng.uniform(0.4, 0.9), offcentre=rng.random() < 0.5)(im, rng)
    if rng.random() < 0.4:
        scale = int(rng.integers(3, 7))
        im = full_res_thin(im, rng, scale=scale, thin=_odd(scale * rng.uniform(0.6, 1.3)))
    if rng.random() < 0.5:
        im = brightness(rng.uniform(0.5, 1.35))(im, rng)
    if rng.random() < 0.4:
        im = uneven_light(rng.uniform(0.35, 0.85))(im, rng)
    if rng.random() < 0.4:
        im = glare(im, rng)
    if rng.random() < 0.3:
        im = blur(rng.uniform(0.002, 0.015))(im, rng)
    if rng.random() < 0.25:
        im = noise(rng.uniform(2, 6))(im, rng)   # phone photos are only mildly noisy
    return im


def swap_colour_random(im, rng):
    """Random marker colour change: random permutation of the RGB channels."""
    a = _arr(im)
    return _img(a[..., rng.permutation(3)])
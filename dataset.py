"""
Dataset loading, caching and augmentation.

Expected layout:
    data/O/*.jpg
    data/X/*.jpg

Every photo goes through the SAME path as in Model Playground:
    playground input (whole photo -> 64x64 gray, -1..1) -> darkness (0 white .. 1 black)
(see playground.py). Results are cached in data/cache_playground.npz.
Delete that file after adding/removing photos or changing playground.py.
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Training images go through the SAME path as the playground: browser 64x64 gray -> darkness.
from playground import SIZE, darkness_from_file as preprocess

CLASSES = ["O", "X"]          # label 0 = O, label 1 = X

# IMPORTANT: your actual folders are data/O and data/X
RAW_DIR = "data"

CACHE = "data/cache_playground.npz"   # separate from the old preprocess.py cache


def load_cached(raw_dir=RAW_DIR, cache=CACHE):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], list(d["paths"])

    X = []
    y = []
    paths = []

    for label, name in enumerate(CLASSES):
        folder = os.path.join(raw_dir, name)

        print(f"Loading class {name} from {folder}")

        for p in sorted(glob.glob(os.path.join(folder, "*"))):
            img = preprocess(p)

            if img is None:
                print("skipped (no ink found):", p)
                continue

            X.append(img)
            y.append(label)
            paths.append(p)

    if len(X) == 0:
        raise RuntimeError(
            "No images were loaded. Make sure data/O and data/X contain image files."
        )

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.savez_compressed(
        cache,
        X=X,
        y=y,
        paths=np.array(paths)
    )

    print(
        f"cached {len(y)} images "
        f"({(y == 0).sum()} O, {(y == 1).sum()} X)"
    )

    return X, y, paths


def group_split(y, paths, val_frac=0.2, seed=42):
    """
    Split by source photo so crops from the same original photo
    cannot appear in both training and validation.

    Consecutive source photos overlap (same board photographed twice),
    so the same drawing can appear in both. Each overlapping set is
    treated as ONE group so both copies always land on the same side.
    """

    import re

    rng = np.random.default_rng(seed)

    SAME_BOARD = {"07": "06", "09": "08", "10": "08",
                  "12": "11", "13": "11", "15": "14", "17": "16"}

    # extract source ID from filenames like crop_0123_src04.jpg,
    # then map overlapping photos to one shared group
    groups = []

    for path in paths:
        match = re.search(r"src(\d+)", path)

        if match:
            src = match.group(1)
            groups.append(SAME_BOARD.get(src, src))
        else:
            raise ValueError(f"Could not find source ID in {path}")

    groups = np.array(groups)

    # shuffle whole groups
    unique_groups = rng.permutation(np.unique(groups))

    n_val_groups = max(
        1,
        int(round(len(unique_groups) * val_frac))
    )

    val_groups = set(unique_groups[:n_val_groups])

    train_indices = []
    val_indices = []

    for i, group in enumerate(groups):
        if group in val_groups:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return (
        np.array(train_indices),
        np.array(val_indices)
    )


class RandomStroke:
    """
    Randomly make marker strokes thicker or thinner.
    """

    def __call__(self, x):
        r = torch.rand(1).item()

        if r < 0.3:
            # thicker
            x = F.max_pool2d(
                x.unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0)

        elif r < 0.5:
            # thinner
            x = -F.max_pool2d(
                -x.unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0)

        return x


AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),

    v2.RandomAffine(
        degrees=20,
        translate=(0.08, 0.08),
        scale=(0.85, 1.1),
        shear=(-15, 15, -15, 15)
    ),

    v2.ElasticTransform(
        alpha=10.0,
        sigma=4.0
    ),

    RandomStroke(),
])
# In RandomStroke, the 3 in both max_pool2d(..., 3, 1, 1) calls can become 5, 1, 2, so the 
# thicker and thinner stroke effects stay about the same size relative to the image


class XODataset(Dataset):
    def __init__(self, X, y, augment=False, copies=1):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y)

        self.augment = augment
        self.copies = copies

    def __len__(self):
        return len(self.y) * self.copies

    def __getitem__(self, i):
        i %= len(self.y)

        x = self.X[i]
        y = self.y[i]

        if self.augment:
            x = AUGMENT(x).clamp(0, 1)

        return x, y
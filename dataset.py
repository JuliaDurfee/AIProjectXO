"""
Dataset loading, caching and augmentation.
 
Expected layout (one shape per photo):
    data/raw/O/*.jpg
    data/raw/X/*.jpg
 
All photos are preprocessed once and cached in data/cache.npz.
Delete that file after adding photos or changing preprocess.py.
"""
import glob
import os
 
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
 
from preprocess import SIZE, preprocess
 
CLASSES = ["O", "X"]          # label 0 = O, label 1 = X
RAW_DIR = "data/raw"
CACHE = "data/cache.npz"
 
 
def load_cached(raw_dir=RAW_DIR, cache=CACHE):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], list(d["paths"])
    X, y, paths = [], [], []
    for label, name in enumerate(CLASSES):
        for p in sorted(glob.glob(os.path.join(raw_dir, name, "*"))):
            img = preprocess(p)
            if img is None:
                print("skipped (no ink found):", p)
                continue
            X.append(img); y.append(label); paths.append(p)
    X, y = np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y, paths=np.array(paths))
    print(f"cached {len(y)} images ({(y == 0).sum()} O, {(y == 1).sum()} X)")
    return X, y, paths
 
 
def stratified_split(y, val_frac=0.2, seed=0):
    """Split BEFORE augmenting so augmented copies of a val image never end up in training."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        n_val = max(1, int(len(idx) * val_frac))
        va += list(idx[:n_val]); tr += list(idx[n_val:])
    return np.array(tr), np.array(va)
 
 
class RandomStroke:
    """Randomly make strokes thicker or thinner (different markers / pressure)."""
    def __call__(self, x):
        r = torch.rand(1).item()
        if r < 0.3:
            x = F.max_pool2d(x.unsqueeze(0), 3, 1, 1).squeeze(0)       # thicker
        elif r < 0.5:
            x = -F.max_pool2d(-x.unsqueeze(0), 3, 1, 1).squeeze(0)     # thinner
        return x
 
 
# Geometric augmentation only: colour/lighting are already removed by preprocessing.
# Keep rotation modest: an X rotated 45 degrees becomes a "+".
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomAffine(degrees=20, translate=(0.08, 0.08), scale=(0.85, 1.1), shear=(-15, 15, -15, 15)),
    v2.ElasticTransform(alpha=12.0, sigma=3.0),
    RandomStroke(),
])
 
 
class XODataset(Dataset):
    def __init__(self, X, y, augment=False, copies=1):
        self.X = torch.from_numpy(X).unsqueeze(1)   # (N, 1, SIZE, SIZE)
        self.y = torch.from_numpy(y)
        self.augment, self.copies = augment, copies
 
    def __len__(self):
        return len(self.y) * self.copies           # 'copies' = extra augmented views per epoch
 
    def __getitem__(self, i):
        i %= len(self.y)
        x = self.X[i]
        if self.augment:
            x = AUGMENT(x).clamp(0, 1)
        return x, self.y[i]
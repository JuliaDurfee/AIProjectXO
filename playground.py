"""
Bridge between this repo and Model Playground (https://supertweety.github.io/model-playground/).

The playground does NOT use our preprocess.py. It sends every model the same input
(contract "ox-gray64-v1"): the WHOLE photo squashed to 64x64, grayscale, black = -1, white = +1.
No crop, no threshold, no colour.

So we move our preprocessing INSIDE the network:

    browser 64x64 gray  ->  ContractToInk (fixed, no weights)  ->  our classifier  ->  logits [O, X]

ContractToInk only uses simple ops (max-pool, min/max reductions, matmul), so it exports to ONNX
and runs in the browser. The same module is used to build the training data (dataset.py), so the
classifier sees exactly the same kind of input in training and in the playground.

Steps inside ContractToInk (64x64 versions of the ideas in preprocess.py):
  1. brightness b in [0,1]
  2. board background = morphological closing of b (max-pool then min-pool, k=11):
     removes the thin dark strokes, keeps shadows and lighting gradients.
     Capped at the image's average brightness + 0.06, so bright ceiling-light
     reflections cannot make the board around them look like ink.
  3. ink = how much darker than the local board a pixel is, relative to board brightness
     (lighting invariant; marker colour only matters through its grayscale darkness)
  4. thicken by 1 px (3x3 max-pool): joins thin strokes that the browser's 64x64
     sampling of a large photo breaks into dots
  5. divide by the average of the 64 strongest ink pixels (not the single max, so one
     heavy dot where the pen stopped cannot make the rest of the stroke look faint),
     then soft threshold (0 below 0.3, 1 above 0.7)
  6. fade out the outer 3 pixels (fragments of neighbouring shapes at the crop edge)
  7. crop to the bounding box of the ink (+12% margin) and stretch back to 64x64,
     done with two 64x64 interpolation matrices (Ry @ img @ Rx^T), then thicken by 1 px
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the playground's own reference loader, so training images get byte-identical
# preprocessing to what the website does. Needs the model-playground submodule.
_PG = Path(__file__).resolve().parent / "model-playground"
sys.path.insert(0, str(_PG))
from python.common import load_image, preprocess as contract_preprocess  # noqa: E402

SIZE = 64


def contract_tensor(path):
    """Photo file -> [1, 64, 64] float32 in [-1, 1], exactly what the browser sends."""
    return contract_preprocess(load_image(str(path)))


class ContractToInk(nn.Module):
    def __init__(self, bg_kernel=11, low=0.3, high=0.7, border=3, pad=0.12, cap=0.06, top_k=64):
        super().__init__()
        self.k, self.low, self.high, self.pad = bg_kernel, low, high, pad
        self.cap, self.top_k = cap, top_k
        idx = torch.arange(SIZE, dtype=torch.float32)
        edge = torch.minimum(idx, SIZE - 1 - idx)
        w = torch.clamp(edge / border, 0, 1)
        self.register_buffer("idx", idx)
        self.register_buffer("t", idx / (SIZE - 1))
        self.register_buffer("border", (w[:, None] * w[None, :])[None, None])

    def _maxpool(self, x):
        p = self.k // 2
        return F.max_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), self.k, stride=1)

    def _range(self, profile):
        """profile [B,1,64] of 0/1 -> first and last index with ink, each [B,1,1]."""
        lo = (self.idx + 1000.0 * (1 - profile)).amin(dim=-1, keepdim=True)
        hi = (self.idx * profile).amax(dim=-1, keepdim=True)
        found = profile.amax(dim=-1, keepdim=True) > 0
        lo = torch.where(found, lo, torch.zeros_like(lo))
        hi = torch.where(found, hi, torch.full_like(hi, SIZE - 1))
        margin = (hi - lo + 1) * self.pad
        return lo - margin, hi + margin

    def _resample_matrix(self, lo, hi):
        src = (lo + (hi - lo) * self.t).unsqueeze(-1)                  # [B,1,64,1]
        return torch.relu(1 - torch.abs(src - self.idx))               # [B,1,64,64]

    def forward(self, x):                                              # x: [B,1,64,64] in [-1,1]
        b = (x + 1) / 2
        bg = -self._maxpool(-self._maxpool(b))                         # closing
        bg = torch.minimum(bg, b.mean(dim=(2, 3), keepdim=True) + self.cap)   # ignore reflections
        ink = torch.relu(bg - b) / (bg + 0.05)
        ink = F.max_pool2d(ink, 3, stride=1, padding=1)               # join dotted strokes
        ref = ink.flatten(1).topk(self.top_k, dim=1).values.mean(dim=1)  # [B]
        ink = ink / (ref.view(-1, 1, 1, 1) + 1e-3)
        m = torch.clamp((ink - self.low) / (self.high - self.low), 0, 1) * self.border

        hard = (m > 0.5).to(m.dtype)
        y0, y1 = self._range(hard.amax(dim=3))
        x0, x1 = self._range(hard.amax(dim=2))
        Ry, Rx = self._resample_matrix(y0, y1), self._resample_matrix(x0, x1)
        out = Ry @ m @ Rx.transpose(-1, -2)
        out = F.max_pool2d(out, 3, stride=1, padding=1)
        return torch.clamp(out, 0, 1)


_TO_INK = ContractToInk().eval()


@torch.no_grad()
def ink_from_file(path):
    """Photo file -> 64x64 float32 numpy ink map, via the browser's input format.
    Drop-in replacement for preprocess.preprocess() in dataset.py."""
    x = contract_tensor(path).unsqueeze(0)
    return _TO_INK(x)[0, 0].numpy().astype(np.float32)


class PlaygroundModel(nn.Module):
    """What gets exported: browser input -> ContractToInk -> classifier -> [O, X] logits."""
    def __init__(self, classifier):
        super().__init__()
        self.to_ink = ContractToInk()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.to_ink(x))


class PerceptronNet(nn.Module):
    """Classifier 1 as a network: the hand-set W from perceptron.py, no training.
    score = sum(W * x) + b; logits = [-score/2, +score/2] so softmax picks X when score > 0."""
    def __init__(self, scale=0.1):
        super().__init__()
        import perceptron
        self.register_buffer("W", torch.from_numpy(perceptron.W.astype(np.float32))[None, None])
        self.bias = float(perceptron.BIAS)
        self.scale = scale  # only makes the shown probabilities less extreme

    def forward(self, x):
        s = (x * self.W).sum(dim=(2, 3)) + self.bias                  # [B,1]
        return torch.cat([-s / 2, s / 2], dim=1) * self.scale
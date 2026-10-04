# X vs O whiteboard classifier

Team mini-project for Intro to AI: three classifiers that decide whether a shape drawn on a
whiteboard (blindfolded, unknown marker colour, unknown lighting) is an **X** or an **O**.
In class, the models run in [Model Playground](https://supertweety.github.io/model-playground/).

| # | Classifier | How it gets its weights | Accuracy on `test_data` |
|---|---|---|---|
| 1 | Single perceptron | Set **by hand** in `perceptron.py`, no training | 99.0% |
| 2 | Multi-layer perceptron (MLP) | Trained on our photos | 100% |
| 3 | Convolutional neural network (CNN) | Trained on our photos | 100% |

All three see exactly the same input the playground sends, so the numbers above are measured
the way the models run in class.

## How it works

The playground gives every model the same input: the **whole photo squashed to 64×64
grayscale** (black = −1, white = +1). It does not crop, threshold or remove colour.

So each exported model starts with a fixed preprocessing layer, `ContractToInk`
(in `playground.py`), followed by the classifier:

```
photo -> playground: 64x64 gray -> ContractToInk -> perceptron / MLP / CNN -> [O, X] scores
```

`ContractToInk` has no trainable weights. It turns the gray image into an **ink map**
(1 = ink, 0 = board) that looks the same whatever the lighting or marker colour:

1. Estimate the board without the strokes (morphological closing), capped near the image's
   average brightness so ceiling-light reflections are ignored.
2. Mark as ink the pixels that are much darker than the board around them
   (lighting- and colour-independent).
3. Thicken by one pixel, which joins thin strokes that the playground's 64×64 sampling of a
   large phone photo breaks into dots.
4. Scale by the 64 strongest ink pixels (so one heavy dot where the pen stopped can't make
   the rest of the stroke look faint), then soft-threshold.
5. Fade out the image border, crop to the ink's bounding box and stretch it back to 64×64,
   so every shape has the same size and position.

The same `ContractToInk` builds the training data, so the networks see the same kind of
input in training as in the playground.

## The classifiers (`models.py`, `perceptron.py`)

- **Perceptron**: one 64×64 weight template, chosen by hand. Ink in the centre and the
  corners counts towards X (an X crosses in the middle and ends in the corners); ink at the
  midpoints of the edges counts towards O. `weights.png` shows the template.
- **MLP**: flattens the 64×64 ink map; layers 4096 → 256 → 64 → 2 with ReLU and dropout 0.3.
- **CNN**: four 3×3 convolution blocks (16, 32, 64, 64 channels) with GroupNorm and ReLU,
  two max-pools, global average pooling, dropout 0.3, then a linear layer to 2 outputs.
  GroupNorm replaced BatchNorm because BatchNorm's running statistics were unstable on our
  small dataset. Its epsilon is 1e-3 so near-blank inputs don't magnify rounding differences
  in the ONNX export.

## Dataset

- `data/O`, `data/X`: 184 O and 259 X crops from 17 whiteboard photos, all drawn by the
  same team member. Files are named `crop_<number>_src<photo>.jpg`.
- `test_data/O`, `test_data/X`: 57 O and 40 X from separate photos. Never used for training
  or for choosing the best epoch.
- **Train/validation split** (`dataset.py`, `group_split`): split by source photo, so crops
  from one photo never end up on both sides. Consecutive source photos overlap (the same board
  photographed twice), so overlapping photos count as one group. We kept those re-photographed
  shapes for their extra lighting and angle variety.
- **Photo-level augmentation** (`photo_aug.py`): when the cache is built, every photo gets
  5 extra random versions *before* ink extraction: different marker colour, faded ink,
  loose or off-centre crop, full-resolution thin strokes, brightness, uneven light, glare,
  blur and noise. That gives 2,658 ink maps from 443 photos.
- **Ink-map augmentation** (`dataset.py`, `AUGMENT`): during training, random flips,
  rotation, shift, scale, shear, elastic distortion and thicker/thinner strokes. Flips are
  safe because a flipped X is still an X and a flipped O is still an O.

## Setup

```
git clone --recurse-submodules <https://github.com/JuliaDurfee/AIProjectXO.git>   # or: git submodule update --init
cd AIProjectXO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r model-playground/python/requirements.txt
python -m pip install "torchvision==0.22.1" opencv-python matplotlib
```

Use Python 3.10–3.12 and `python -m pip` inside the environment. The `model-playground`
folder is a git submodule; it provides the playground's image loader and export helper.

## Usage

```
python train.py --model mlp --patience 15
python train.py --model cnn --patience 15 --lr 0.0005
python stress_test.py           # robustness check (optional)
python export_playground.py     # writes onnx/perceptron.onnx, onnx/mlp.onnx, onnx/cnn.onnx
```

The first training run builds `data/cache_playground_aug.npz`, which takes a few minutes.
**Delete that file** after adding or removing photos, or after changing `playground.py` or
`photo_aug.py`.

`export_playground.py` first prints each model's accuracy on `test_data`, then exports it
and checks that the ONNX file gives the same outputs as PyTorch.

## In class

1. Open [Model Playground](https://supertweety.github.io/model-playground/) on the phone and
   choose one of the `.onnx` files from `onnx/`.
2. Take a photo of one shape and **crop it square and tight**: the shape fills about
   60–70% of the frame, one shape only, no glare near it.
3. Select **Run model**. The scores are in O, X order.

## Robustness check (`stress_test.py`)

Alters every test photo in ways class conditions may differ and runs all three models
through the playground's exact input path. Most recent results:

| Condition | Perceptron | MLP | CNN |
|---|---|---|---|
| Each condition on its own (lighting, faded ink, other colour, blur, noise, glare, loose crops, 3:4 photo, thin strokes) | 98–100% | 99–100% | 100% |
| Classroom combo (loose crop + thin strokes + uneven light + glare) | 92.8% | 95.9% | 95.9% |

Before photo-level augmentation, the CNN scored 80.4% on the classroom combo.

## Files

| File | Purpose |
|---|---|
| `playground.py` | `ContractToInk`, the playground's input format, model wrappers for export |
| `perceptron.py` | Classifier 1: the hand-set weight template |
| `models.py` | Classifiers 2 and 3: MLP and CNN |
| `dataset.py` | Loading, caching, photo- and ink-level augmentation, grouped split |
| `photo_aug.py` | Photo alterations used for augmentation and the stress test |
| `train.py` | Trains the MLP or CNN, early stopping on validation loss |
| `export_playground.py` | Test accuracy + ONNX export for the playground |
| `playground_export_loose.py` | Copy of the playground's export helper with a slightly looser PyTorch/ONNX tolerance (the CNN differs by ~0.0001 because of floating-point rounding order) |
| `stress_test.py` | Robustness check under simulated classroom conditions |
| `preprocess.py` | Original preprocessing; still provides `SIZE` and is used by `perceptron.py` |
| `crop_images.py`, `contact_sheet.py`, `manifest.csv` | Building the dataset: cropping shapes out of whiteboard photos and reviewing them |
| `models/` | Trained weights (`mlp.pt`, `cnn.pt`) |
| `onnx/` | Exported models for the playground |
| `model-playground/` | Git submodule: the playground's loader and export helper |
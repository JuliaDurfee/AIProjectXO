X vs O whiteboard classifier
pip install -r requirements.txt
python preprocess.py data/raw/X/*.jpg     # check debug/*_steps.png looks right
python perceptron.py                      # classifier 1 (hand-set weights), also builds data/cache.npz
python train.py --model mlp               # classifier 2
python train.py --model cnn               # classifier 3
python predict.py new_photo.jpg           # all three, for use in class

Files: preprocess.py (photo -> 28x28 ink map), perceptron.py, dataset.py (loading + augmentation), models.py (MLP, CNN), train.py, predict.py.

Data layout: data/raw/O/*.jpg, data/raw/X/*.jpg, one shape per photo. Delete data/cache.npz after adding photos or changing preprocessing.
"""
Optional: build data/mnist.npz from the raw MNIST files, if the automatic OpenML
download in fed_kmeans_shapley.py does not work on your network.

1. Download these two files (from https://github.com/fgnt/mnist or any MNIST mirror)
   into a folder called data/ next to this script:
       train-images-idx3-ubyte.gz
       train-labels-idx1-ubyte.gz
2. Run:  python prepare_mnist.py
"""
import gzip, os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
d = os.path.join(here, "data")
X = np.frombuffer(gzip.open(os.path.join(d, "train-images-idx3-ubyte.gz")).read(), np.uint8, offset=16).reshape(-1, 784)
y = np.frombuffer(gzip.open(os.path.join(d, "train-labels-idx1-ubyte.gz")).read(), np.uint8, offset=8).astype(int)
assert X.shape == (60000, 784) and y.shape == (60000,)
np.savez_compressed(os.path.join(d, "mnist.npz"), X=X, y=y)
print("saved data/mnist.npz:", X.shape, np.bincount(y))

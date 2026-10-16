"""Python cross-check of src/classify_nn_knn.m.

Runs the same NN / KNN classification on the feature files in ../features
without needing MATLAB. Tie-breaking matches MATLAB: a stable sort for
distances, and the smallest class index wins a tied KNN vote (like mode).

Usage (from the repo root):
    pip install numpy scipy
    python scripts/evaluate.py
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FEATURE_DIR = Path(__file__).resolve().parent.parent / "features"
K = 5
CLASSES = ["sunflower", "crab", "chair", "cellphone", "dolphin",
           "stopsign", "helicopter", "lamp", "kangaroo", "laptop"]


def load_features(path):
    """Return an N x D matrix, whatever the variable is called, minus zero rows."""
    data = loadmat(path)
    for key, value in data.items():
        if (not key.startswith("__") and isinstance(value, np.ndarray)
                and value.ndim == 3 and value.shape[0] == 1):
            feats = value[0].T.astype(float)
            return feats[np.abs(feats).sum(axis=1) > 0]
    raise ValueError(f"No 1 x D x N feature array in {path}")


def main():
    x_train, y_train, x_test, y_test = [], [], [], []
    for c, name in enumerate(CLASSES):
        tr = load_features(FEATURE_DIR / f"{name}_featurevector.mat")
        te = load_features(FEATURE_DIR / f"{name}Test_featurevector.mat")
        x_train.append(tr); y_train += [c] * len(tr)
        x_test.append(te); y_test += [c] * len(te)
    x_train, x_test = np.vstack(x_train), np.vstack(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    print(f"{len(y_train)} training images, {len(y_test)} test images, "
          f"{len(CLASSES)} classes\n")

    dist = np.linalg.norm(x_test[:, None, :] - x_train[None, :, :], axis=2)
    order = np.argsort(dist, axis=1, kind="stable")
    pred_nn = y_train[order[:, 0]]
    pred_knn = np.array([np.argmax(np.bincount(y_train[row[:K]],
                                               minlength=len(CLASSES)))
                         for row in order])

    print("Overall accuracy")
    print(f"  NN       : {100 * np.mean(pred_nn == y_test):5.1f} %")
    print(f"  KNN (K={K}): {100 * np.mean(pred_knn == y_test):5.1f} %\n")
    print(f"{'Class':12s} {'Tests':>6s} {'NN':>8s} {'KNN':>8s}")
    for c, name in enumerate(CLASSES):
        idx = y_test == c
        print(f"{name:12s} {idx.sum():6d} {100 * np.mean(pred_nn[idx] == c):7.1f}%"
              f" {100 * np.mean(pred_knn[idx] == c):7.1f}%")


if __name__ == "__main__":
    main()

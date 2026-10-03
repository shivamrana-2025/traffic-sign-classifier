"""Dataset loading for GTSRB (Kaggle layout: Train/<class_id>/*.png, Test/, Test.csv)."""
import cv2
import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_DIR, NUM_CLASSES


def _read(path, size):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_train(size):
    cache = CACHE_DIR / f"train_{size}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    train_dir = DATA_DIR / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"{train_dir} not found. Download GTSRB from Kaggle and extract it into data/gtsrb/ "
            "(see README)."
        )
    X, y = [], []
    for class_id in range(NUM_CLASSES):
        for p in sorted((train_dir / str(class_id)).glob("*.png")):
            X.append(_read(p, size))
            y.append(class_id)
    X, y = np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)
    np.savez(cache, X=X, y=y)
    return X, y


def load_test(size):
    cache = CACHE_DIR / f"test_{size}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    df = pd.read_csv(DATA_DIR / "Test.csv")
    X = np.array([_read(DATA_DIR / p, size) for p in df["Path"]], dtype=np.uint8)
    y = df["ClassId"].to_numpy(dtype=np.int64)
    np.savez(cache, X=X, y=y)
    return X, y

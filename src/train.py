"""Train a model.

    python -m src.train --model baseline
    python -m src.train --model mobilenet
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .config import MODEL_SIZES, MODELS_DIR, RESULTS_DIR, SEED
from .data import load_train
from .models import build_baseline, build_mobilenet, unfreeze_top


def callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def merge(histories):
    out = {}
    for h in histories:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(float(i) for i in v)
    return out


def plot_history(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["accuracy"], label="train")
    ax[0].plot(hist["val_accuracy"], label="val")
    ax[0].set_title(f"{title} - accuracy")
    ax[1].plot(hist["loss"], label="train")
    ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title(f"{title} - loss")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SIZES), required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    size = MODEL_SIZES[args.model]

    X, y = load_train(size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"train={len(X_tr)} val={len(X_val)} size={size}")

    fit_kw = dict(validation_data=(X_val, y_val), batch_size=args.batch_size, callbacks=callbacks())

    if args.model == "baseline":
        model = build_baseline(size)
        compile_model(model, 1e-3)
        model.summary()
        histories = [model.fit(X_tr, y_tr, epochs=args.epochs, **fit_kw)]
    else:
        model, base = build_mobilenet(size)
        compile_model(model, 1e-3)
        model.summary()
        print("\n--- Phase 1: training the classification head ---")
        histories = [model.fit(X_tr, y_tr, epochs=args.epochs, **fit_kw)]
        print("\n--- Phase 2: fine-tuning top MobileNetV2 layers ---")
        unfreeze_top(base)
        compile_model(model, 1e-4)  # recompile after changing trainable flags
        histories.append(model.fit(X_tr, y_tr, epochs=args.finetune_epochs, **fit_kw))

    hist = merge(histories)
    model.save(MODELS_DIR / f"{args.model}.keras")
    with open(RESULTS_DIR / f"{args.model}_history.json", "w") as f:
        json.dump(hist, f)
    plot_history(hist, RESULTS_DIR / f"{args.model}_curves.png", args.model)
    print(f"Saved model to models/{args.model}.keras")


if __name__ == "__main__":
    main()

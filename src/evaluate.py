"""Evaluate a trained model on the official GTSRB test set.

    python -m src.evaluate --model baseline
    python -m src.evaluate --model mobilenet

Outputs (in results/): metrics json, classification report, confusion matrix,
misclassified examples, and a comparison table across all evaluated models.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras

from .config import MODEL_SIZES, MODELS_DIR, NUM_CLASSES, RESULTS_DIR
from .data import load_test
from .labels import CLASS_NAMES


def plot_confusion(cm, path, name):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_title(f"Confusion matrix (row-normalised) - {name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(0, NUM_CLASSES, 2))
    ax.set_yticks(range(0, NUM_CLASSES, 2))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_confusions(cm, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:k], off.shape))[0]
    return [(int(t), int(p), int(off[t, p])) for t, p in idx if off[t, p] > 0]


def plot_errors(X, y_true, y_pred, path, n=16):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, pick):
        a.imshow(X[i])
        a.set_title(
            f"true: {CLASS_NAMES[y_true[i]][:22]}\npred: {CLASS_NAMES[y_pred[i]][:22]}",
            fontsize=8, color="crimson",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_comparison():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*_metrics.json"))]
    if not rows:
        return
    lines = [
        "| Model | Input size | Params | Model size (MB) | Test accuracy | Macro F1 | Latency (ms/img) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        lines.append(
            f"| {m['model']} | {m['input_size']}x{m['input_size']} | {m['params']:,} | "
            f"{m['size_mb']:.1f} | {m['accuracy'] * 100:.2f}% | {m['macro_f1']:.4f} | {m['latency_ms']:.2f} |"
        )
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SIZES), required=True)
    args = ap.parse_args()

    size = MODEL_SIZES[args.model]
    model_path = MODELS_DIR / f"{args.model}.keras"
    model = keras.models.load_model(model_path)
    X, y = load_test(size)

    t0 = time.perf_counter()
    probs = model.predict(X, batch_size=128, verbose=0)
    latency_ms = (time.perf_counter() - t0) / len(X) * 1000
    y_pred = probs.argmax(1)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"{args.model}: test accuracy = {acc:.4f}, macro F1 = {f1:.4f}")

    report = classification_report(
        y, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0
    )
    (RESULTS_DIR / f"{args.model}_report.txt").write_text(report)

    cm = confusion_matrix(y, y_pred, labels=list(range(NUM_CLASSES)))
    plot_confusion(cm, RESULTS_DIR / f"{args.model}_confusion_matrix.png", args.model)
    plot_errors(X, y, y_pred, RESULTS_DIR / f"{args.model}_errors.png")

    print("\nMost confused pairs (true -> predicted : count)")
    for t, p, c in top_confusions(cm):
        print(f"  {CLASS_NAMES[t]}  ->  {CLASS_NAMES[p]} : {c}")

    metrics = {
        "model": args.model,
        "input_size": size,
        "params": int(model.count_params()),
        "size_mb": model_path.stat().st_size / 1e6,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "latency_ms": float(latency_ms),
    }
    (RESULTS_DIR / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_comparison()


if __name__ == "__main__":
    main()

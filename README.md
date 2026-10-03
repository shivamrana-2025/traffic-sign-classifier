# 🚦 Traffic Sign Classification (GTSRB)

Deep-learning classifier that recognises **43 types of German traffic signs**, with two models compared head-to-head:

1. **Baseline CNN** trained from scratch
2. **MobileNetV2** with transfer learning (head training, then fine-tuning)

Includes evaluation (confusion matrix, error analysis) and a **Streamlit web app** for live predictions.

<!-- Add a GIF of the app here: ![demo](results/demo.gif) -->

## Results (official GTSRB test set, 12,630 images)

> Run the pipeline and paste your numbers here (`results/comparison.md` is generated automatically).

| Model | Input size | Params | Model size (MB) | Test accuracy | Macro F1 | Latency (ms/img) |
|---|---|---|---|---|---|---|
| baseline | 48x48 | - | - | - | - | - |
| mobilenet | 96x96 | - | - | - | - | - |

Plots saved in `results/`: training curves, confusion matrix, misclassified examples.

## Project structure

```
├── app.py                 # Streamlit demo
├── src/
│   ├── config.py          # paths & constants
│   ├── data.py            # dataset loading + caching
│   ├── labels.py          # 43 class names
│   ├── models.py          # baseline CNN + MobileNetV2
│   ├── train.py           # training script
│   └── evaluate.py        # metrics, confusion matrix, error analysis
├── data/                  # put dataset here (git-ignored)
├── models/                # saved models (git-ignored)
├── results/               # plots & metrics
└── notebooks/             # EDA
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dataset:** download [GTSRB from Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeow/gtsrb-german-traffic-sign) and extract so the layout is:

```
data/gtsrb/
├── Train/0 ... Train/42/*.png
├── Test/*.png
└── Test.csv
```

## Run

```bash
python -m src.train --model baseline
python -m src.train --model mobilenet
python -m src.evaluate --model baseline
python -m src.evaluate --model mobilenet
streamlit run app.py
```

Use `--epochs` / `--finetune-epochs` / `--batch-size` to tweak training. A GPU (or Colab) helps a lot for MobileNetV2.

## Design decisions

- **Augmentation** (rotation, zoom, shift, contrast) runs inside the model, active only at training time. **No horizontal flips**, since "turn left" and "turn right" would swap meaning.
- **Preprocessing lives in the model**, so the app just feeds raw pixels.
- **Two-phase transfer learning**: train the new head with the base frozen, then fine-tune the top 30 layers at a low learning rate with BatchNorm kept frozen.
- **Honest evaluation**: final numbers come from the official, separate test set, not the validation split.
- **Known caveat**: GTSRB training images come from video tracks (about 30 near-identical frames per sign), so a random validation split is slightly optimistic. The test set is unaffected.

## Possible extensions

- Detect signs in full road scenes (YOLO), then classify each crop
- Export to TFLite for mobile/edge deployment
- Grad-CAM visualisations to show what the model looks at

## Tech

Python · TensorFlow/Keras · OpenCV · scikit-learn · Matplotlib · Streamlit

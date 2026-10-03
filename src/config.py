from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gtsrb"      # extracted Kaggle dataset goes here
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

NUM_CLASSES = 43
SEED = 42

# input resolution used by each model
MODEL_SIZES = {"baseline": 48, "mobilenet": 96}

for _d in (CACHE_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

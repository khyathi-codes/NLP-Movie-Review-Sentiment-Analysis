# config.py  -- shared settings (owner: Member 1). Everyone imports from here.
import os

RANDOM_STATE = 42
SAMPLE_SIZE = 10000      # rows to use (None = use all 50k; slower)
TEST_SIZE = 0.2
NLP_SAMPLE = 400         # reviews used for POS / parsing / NER analysis

DATA_DIR, OUT_DIR, MODEL_DIR = "data", "outputs", "models"
for d in (DATA_DIR, OUT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

RAW_CSV = os.path.join(DATA_DIR, "IMDB Dataset.csv")   # Kaggle file (optional)
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "clean.csv")

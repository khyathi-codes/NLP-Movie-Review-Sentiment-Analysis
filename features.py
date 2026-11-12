# features.py  (Member 3)  Feature engineering -> data/hand_*.npy
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from config import *

nltk.download("vader_lexicon", quiet=True)
SIA = SentimentIntensityAnalyzer()
NAMES = ["review_length", "exclamation_count", "question_count", "negation_count",
         "vader_negative", "vader_neutral", "vader_positive", "vader_compound"]


def handcrafted(raw):
    text = re.sub(r"<[^>]+>", " ", str(raw))
    words = text.split()
    n = max(len(words), 1)
    v = SIA.polarity_scores(text)
    return [len(words), text.count("!"), text.count("?"),
            len(re.findall(r"\b(not|no|never)\b|n't", text.lower())),
            v["neg"], v["neu"], v["pos"], v["compound"]]


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV)
    train, test = df[df.split == "train"], df[df.split == "test"]
    F_tr = np.array([handcrafted(t) for t in train.review])
    F_te = np.array([handcrafted(t) for t in test.review])

    sc = StandardScaler().fit(F_tr)                       # fit on train only
    np.save(f"{DATA_DIR}/hand_train.npy", sc.transform(F_tr))
    np.save(f"{DATA_DIR}/hand_test.npy", sc.transform(F_te))
    joblib.dump(sc, f"{MODEL_DIR}/hand_scaler.joblib")

    fd = pd.DataFrame(F_tr, columns=NAMES)
    print("Handcrafted training features:", F_tr.shape, "| testing:", F_te.shape)
    print("Feature names:", NAMES)
    print("\nFeature correlation matrix:\n", fd.corr().round(2))
    print("\nCorrelation of each feature with label:\n",
          fd.corrwith(train.label.reset_index(drop=True)).round(3))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    plt.figure(figsize=(10, 7))
    sns.heatmap(fd.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_correlation.png", dpi=150)

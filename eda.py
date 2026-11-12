# eda.py  (Member 1)  Data analysis -> outputs/eda_*.png
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from config import *

df = pd.read_csv(REVIEWS_CSV)
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print(df["sentiment"].value_counts())

df["n_words"] = df["review"].str.split().str.len()
print(df.groupby("sentiment")["n_words"].describe())

# 1. class balance
df["sentiment"].value_counts().plot(kind="bar", color=["tab:green", "tab:red"])
plt.title("Class distribution"); plt.ylabel("Reviews"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_class_balance.png", dpi=150); plt.close()

# 2. review length distribution
for s, g in df.groupby("sentiment"):
    plt.hist(g["n_words"].clip(upper=800), bins=40, alpha=0.6, label=s)
plt.legend(); plt.title("Review length (words)"); plt.xlabel("Words"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_length.png", dpi=150); plt.close()

# 3. top words per class
cv = CountVectorizer(stop_words="english", max_features=3000)
for s, g in df.groupby("sentiment"):
    X = cv.fit_transform(g["review"].str.replace("<br />", " "))
    freq = pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out())
    top = freq.sort_values(ascending=False).head(15)
    top[::-1].plot(kind="barh", title=f"Top words - {s}")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/eda_top_words_{s}.png", dpi=150)
    plt.close()
    print(f"\nTop words ({s}):", list(top.index))
print("EDA figures saved in", OUT_DIR)

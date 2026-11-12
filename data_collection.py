# data_collection.py  (Member 1)  ->  data/reviews.csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *


def load_raw():
    if os.path.exists(RAW_CSV):                      # Kaggle IMDB 50K dataset
        print("Using Kaggle IMDB dataset")
        df = pd.read_csv(RAW_CSV)[["review", "sentiment"]]
    else:                                            # automatic fallback (2000 reviews)
        print("Kaggle file not found -> using NLTK movie_reviews corpus")
        import nltk
        nltk.download("movie_reviews", quiet=True)
        from nltk.corpus import movie_reviews
        rows = [(movie_reviews.raw(f), c) for c in movie_reviews.categories()
                for f in movie_reviews.fileids(c)]
        df = pd.DataFrame(rows, columns=["review", "sentiment"])
        df["sentiment"] = df["sentiment"].map({"pos": "positive", "neg": "negative"})
    return df


if __name__ == "__main__":
    df = load_raw()
    df = df.dropna().drop_duplicates(subset="review")
    df["label"] = (df["sentiment"] == "positive").astype(int)   # 1=positive, 0=negative
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df, _ = train_test_split(df, train_size=SAMPLE_SIZE,
                                 stratify=df["label"], random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    df.to_csv(REVIEWS_CSV, index=False)
    print("Saved", REVIEWS_CSV, df.shape)
    print(df["sentiment"].value_counts())

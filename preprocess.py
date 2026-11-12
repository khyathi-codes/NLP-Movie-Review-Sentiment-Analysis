# preprocess.py  (Member 2)  ->  data/clean.csv (adds clean_text + train/test split)
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from config import *

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

NEGATIONS = {"no", "nor", "not", "never"}            # keep: they flip sentiment
STOP = set(stopwords.words("english")) - NEGATIONS
LEMMA = WordNetLemmatizer()


def clean_text(text):
    """raw review -> lowercase, no html/urls/punctuation, no stopwords, lemmatized."""
    text = re.sub(r"<[^>]+>", " ", str(text))          # html tags like <br />
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # urls
    text = text.lower()
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)               # didn't -> did not
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    tokens = [t for t in text.split() if (t in NEGATIONS or len(t) > 2)]
    tokens = [LEMMA.lemmatize(t) for t in tokens if t not in STOP]
    return " ".join(tokens)


if __name__ == "__main__":
    df = pd.read_csv(REVIEWS_CSV)
    df["clean_text"] = df["review"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    tr, te = train_test_split(df.index, test_size=TEST_SIZE,
                              stratify=df["label"], random_state=RANDOM_STATE)
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    df.to_csv(CLEAN_CSV, index=False)

    print("BEFORE:", df["review"][0][:200])
    print("AFTER :", df["clean_text"][0][:200])
    print(df["split"].value_counts())

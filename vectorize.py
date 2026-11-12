# vectorize.py  (Member 2)  Bag-of-Words + TF-IDF -> data/*.npz, models/*.joblib
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config import *

df = pd.read_csv(CLEAN_CSV)
train, test = df[df.split == "train"], df[df.split == "test"]

# --- tiny demo for the report: BoW vs TF-IDF on 3 sentences
demo = ["good movie good plot", "bad movie bad acting", "good acting"]
d_bow = CountVectorizer().fit(demo)
print("BoW vocab :", d_bow.get_feature_names_out())
print(d_bow.transform(demo).toarray())
d_tf = TfidfVectorizer().fit(demo)
print(np.round(d_tf.transform(demo).toarray(), 2))

# --- real vectorizers: FIT ON TRAIN ONLY (avoids data leakage)
bow = CountVectorizer(max_features=5000)
Xb_tr, Xb_te = bow.fit_transform(train.clean_text), bow.transform(test.clean_text)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True)
Xt_tr, Xt_te = tfidf.fit_transform(train.clean_text), tfidf.transform(test.clean_text)

for name, m in [("bow_train", Xb_tr), ("bow_test", Xb_te),
                ("tfidf_train", Xt_tr), ("tfidf_test", Xt_te)]:
    save_npz(f"{DATA_DIR}/{name}.npz", m)
joblib.dump(bow, f"{MODEL_DIR}/bow.joblib")
joblib.dump(tfidf, f"{MODEL_DIR}/tfidf.joblib")
print("BoW shape:", Xb_tr.shape, "| TF-IDF shape:", Xt_tr.shape)

# --- top TF-IDF terms per class
terms = np.array(tfidf.get_feature_names_out())
for lab, nm in [(1, "positive"), (0, "negative")]:
    mean = np.asarray(Xt_tr[(train.label == lab).values].mean(axis=0)).ravel()
    print(f"Top TF-IDF terms ({nm}):", list(terms[mean.argsort()[::-1][:15]]))

# classify.py  (Member 5)  ML classification + model comparison -> outputs/model_comparison.csv
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *

df = pd.read_csv(CLEAN_CSV)
y_tr = df[df.split == "train"].label.values
y_te = df[df.split == "test"].label.values

L = lambda n: load_npz(f"{DATA_DIR}/{n}.npz")
H_tr, H_te = np.load(f"{DATA_DIR}/hand_train.npy"), np.load(f"{DATA_DIR}/hand_test.npy")
W_tr, W_te = np.load(f"{DATA_DIR}/w2v_train.npy"), np.load(f"{DATA_DIR}/w2v_test.npy")
T_tr, T_te = L("tfidf_train"), L("tfidf_test")

HW = 0.05  # weight of hand-crafted features (TF-IDF values are small, so scale them down)
FEATURES = {
    "BoW":         (L("bow_train"), L("bow_test")),
    "TF-IDF":      (T_tr, T_te),
    "TF-IDF+Hand": (hstack([T_tr, csr_matrix(H_tr * HW)]).tocsr(),
                    hstack([T_te, csr_matrix(H_te * HW)]).tocsr()),
    "W2V(CBOW)":   (W_tr, W_te),
    "W2V+Hand":    (np.hstack([W_tr, H_tr]), np.hstack([W_te, H_te])),
}
MODELS = {
    "Naive Bayes":         lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Linear SVM":          lambda: LinearSVC(max_iter=5000),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                          random_state=RANDOM_STATE),
}
PLAN = {   # NB needs non-negative features, RF is slow on sparse text
    "BoW":         ["Naive Bayes", "Logistic Regression", "Linear SVM"],
    "TF-IDF":      ["Naive Bayes", "Logistic Regression", "Linear SVM"],
    "TF-IDF+Hand": ["Logistic Regression", "Linear SVM"],
    "W2V(CBOW)":   ["Logistic Regression", "Linear SVM", "Random Forest"],
    "W2V+Hand":    ["Logistic Regression", "Linear SVM", "Random Forest"],
}

rows = []
for feat, model_names in PLAN.items():
    Xtr, Xte = FEATURES[feat]
    for mn in model_names:
        t0 = time.time()
        clf = MODELS[mn]().fit(Xtr, y_tr)
        p = clf.predict(Xte)
        rows.append({"features": feat, "model": mn,
                     "accuracy": accuracy_score(y_te, p),
                     "precision": precision_score(y_te, p),
                     "recall": recall_score(y_te, p),
                     "f1": f1_score(y_te, p),
                     "seconds": round(time.time() - t0, 1)})
        print(rows[-1])

res = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
res.round(4).to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print("\n", res.round(4).to_string())

res["name"] = res["features"] + " + " + res["model"]
res.sort_values("f1").plot(kind="barh", x="name", y="f1", legend=False, figsize=(8, 6))
plt.xlim(res.f1.min() - 0.02, 1); plt.title("Model comparison (F1)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
print("\nBEST:", res.loc[0, "name"])

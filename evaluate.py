# evaluate.py  (Member 5)  Final model, tuning, evaluation -> models/final_model.joblib
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay)
from config import *

df = pd.read_csv(CLEAN_CSV)
train, test = df[df.split == "train"], df[df.split == "test"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                              sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000)),
])
grid = GridSearchCV(pipe, {"clf__C": [0.5, 1, 5, 10]}, cv=3, scoring="f1", n_jobs=-1)
grid.fit(train.clean_text, train.label)
model = grid.best_estimator_
print("Best params:", grid.best_params_, "| CV F1:", round(grid.best_score_, 4))

pred = model.predict(test.clean_text)
proba = model.predict_proba(test.clean_text)[:, 1]
print(classification_report(test.label, pred, target_names=["negative", "positive"]))
print("ROC-AUC:", round(roc_auc_score(test.label, proba), 4))

ConfusionMatrixDisplay.from_predictions(test.label, pred,
                                        display_labels=["negative", "positive"])
plt.title("Confusion matrix"); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
RocCurveDisplay.from_predictions(test.label, proba)
plt.title("ROC curve"); plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150); plt.close()

wrong = test[pred != test.label]
print("\nMisclassified examples:")
for _, r in wrong.head(5).iterrows():
    print(f"- true={r.sentiment}: {r.review[:150]!r}")

joblib.dump(model, f"{MODEL_DIR}/final_model.joblib")
print("Saved models/final_model.joblib")

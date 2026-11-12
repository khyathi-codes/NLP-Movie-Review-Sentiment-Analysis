# embeddings.py  (Member 3)  Word2Vec (CBOW + Skip-gram) -> data/w2v_*.npy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from config import *

df = pd.read_csv(CLEAN_CSV)
train, test = df[df.split == "train"], df[df.split == "test"]
sentences = [t.split() for t in train.clean_text]

cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=2,
                sg=0, workers=4, epochs=10, seed=RANDOM_STATE)      # sg=0 -> CBOW
skip = Word2Vec(sentences, vector_size=100, window=5, min_count=2,
                sg=1, workers=4, epochs=10, seed=RANDOM_STATE)      # sg=1 -> Skip-gram
cbow.save(f"{MODEL_DIR}/word2vec_cbow.model")
skip.save(f"{MODEL_DIR}/word2vec_skipgram.model")
print("Vocabulary size:", len(cbow.wv))

for w in ["movie", "good", "bad", "actor", "boring"]:
    if w in cbow.wv:
        print(f"CBOW  similar to '{w}':", [x for x, _ in cbow.wv.most_similar(w, topn=6)])
        print(f"SKIP  similar to '{w}':", [x for x, _ in skip.wv.most_similar(w, topn=6)])


def doc_vector(text, model=cbow):
    """average of the word vectors of a review"""
    vecs = [model.wv[w] for w in text.split() if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


np.save(f"{DATA_DIR}/w2v_train.npy", np.vstack(train.clean_text.map(doc_vector)))
np.save(f"{DATA_DIR}/w2v_test.npy", np.vstack(test.clean_text.map(doc_vector)))

# PCA plot of sentiment words
words = [w for w in ["good", "great", "excellent", "love", "amazing", "best", "perfect",
                     "bad", "worst", "awful", "boring", "terrible", "waste", "poor",
                     "actor", "director", "plot", "story", "music", "scene"] if w in cbow.wv]
xy = PCA(n_components=2).fit_transform([cbow.wv[w] for w in words])
plt.figure(figsize=(7, 5))
plt.scatter(xy[:, 0], xy[:, 1])
for (x, y), w in zip(xy, words):
    plt.annotate(w, (x, y))
plt.title("Word2Vec (CBOW) - PCA of words")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/word2vec_pca.png", dpi=150)
print("Saved doc vectors + PCA plot")

# PCA of whole-review vectors, coloured by sentiment (as in M3's Colab graph)
D = np.load(f"{DATA_DIR}/w2v_train.npy")
xy = PCA(n_components=2).fit_transform(D)
plt.figure(figsize=(7, 5))
for lab, nm, c in [(0, "negative", "tab:red"), (1, "positive", "tab:green")]:
    m = (train.label == lab).values
    plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, c=c, label=nm)
plt.legend(); plt.title("Word2Vec CBOW review embeddings - PCA")
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/word2vec_reviews_pca.png", dpi=150)

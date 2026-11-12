# nlp_analysis.py  (Member 4)  POS tagging, dependency parsing, NER, semantic labeling,
# aspect-based opinion mining.   Needs: python -m spacy download en_core_web_sm
import re
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import *

nltk.download("vader_lexicon", quiet=True)
SIA = SentimentIntensityAnalyzer()

ASPECTS = {                                   # aspect category -> trigger words
    "acting":    {"acting", "actor", "actress", "cast", "performance", "role"},
    "plot":      {"plot", "story", "script", "screenplay", "writing", "ending"},
    "direction": {"director", "direction", "directing"},
    "visuals":   {"visual", "cinematography", "effect", "photography", "animation"},
    "music":     {"music", "soundtrack", "score", "song"},
}
WORD2ASPECT = {w: a for a, ws in ASPECTS.items() for w in ws}


def get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def polarity(word, negated=False):
    p = SIA.polarity_scores(word)["compound"]
    return -p if negated else p


def extract_opinions(doc):
    """dependency-based (aspect, opinion, polarity) triples."""
    out = []
    for tok in doc:
        if tok.pos_ != "ADJ":
            continue
        neg = any(c.dep_ == "neg" for c in tok.children)
        if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):   # "great acting"
            out.append((tok.head.lemma_.lower(), tok.lemma_.lower(),
                        polarity(tok.lemma_, neg)))
        elif tok.dep_ == "acomp":                                       # "plot was dull"
            verb = tok.head
            neg = neg or any(c.dep_ == "neg" for c in verb.children)
            for s in verb.children:
                if s.dep_ == "nsubj" and s.pos_ in ("NOUN", "PROPN"):
                    out.append((s.lemma_.lower(), tok.lemma_.lower(),
                                polarity(tok.lemma_, neg)))
    return out


def svo_triples(doc):
    """simple semantic roles: agent (nsubj) - predicate (verb) - patient (dobj)."""
    res = []
    for tok in doc:
        if tok.pos_ == "VERB":
            s = [c.text for c in tok.children if c.dep_ == "nsubj"]
            o = [c.text for c in tok.children if c.dep_ == "dobj"]
            if s and o:
                res.append((s[0], tok.lemma_, o[0]))
    return res


if __name__ == "__main__":
    nlp = get_nlp()
    df = pd.read_csv(REVIEWS_CSV).sample(NLP_SAMPLE, random_state=RANDOM_STATE)
    texts = [re.sub(r"<[^>]+>", " ", t)[:1500] for t in df.review]
    docs = list(nlp.pipe(texts, batch_size=32))

    # ---- 1. POS tagging
    pos_by = {0: Counter(), 1: Counter()}
    adj_by = {0: Counter(), 1: Counter()}
    for doc, lab in zip(docs, df.label):
        for t in doc:
            pos_by[lab][t.pos_] += 1
            if t.pos_ == "ADJ" and not t.is_stop:
                adj_by[lab][t.lemma_.lower()] += 1
    pos_df = pd.DataFrame(pos_by).fillna(0).rename(columns={0: "negative", 1: "positive"})
    print("POS counts:\n", pos_df.sort_values("positive", ascending=False).head(8))
    print("Top adjectives (positive):", adj_by[1].most_common(12))
    print("Top adjectives (negative):", adj_by[0].most_common(12))
    pos_df.sort_values("positive", ascending=False).head(8).plot(kind="bar")
    plt.title("POS tag counts by sentiment"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/nlp_pos.png", dpi=150); plt.close()

    # ---- 2. Dependency parsing (one example sentence)
    ex = nlp("The acting was great but the plot was not interesting.")
    print("\nDependency parse:")
    for t in ex:
        print(f"{t.text:12} {t.pos_:6} {t.dep_:10} head={t.head.text}")

    # ---- 3. NER
    ents = Counter()
    people = Counter()
    for doc in docs:
        for e in doc.ents:
            ents[e.label_] += 1
            if e.label_ == "PERSON":
                people[e.text] += 1
    print("\nEntity types:", ents.most_common())
    print("Most mentioned people:", people.most_common(10))

    # ---- 4. Semantic labeling (agent - action - patient)
    print("\nSVO examples:", [t for d in docs[:30] for t in svo_triples(d)][:10])

    # ---- 5. Opinion mining (aspect-based)
    rows = []
    for doc, lab in zip(docs, df.label):
        for aspect, opinion, pol in extract_opinions(doc):
            rows.append({"aspect": aspect, "opinion": opinion, "polarity": pol,
                         "category": WORD2ASPECT.get(aspect, "other"), "label": lab})
    op = pd.DataFrame(rows)
    op = op[op.polarity != 0]            # keep only real opinion words (VADER lexicon)
    op.to_csv(f"{OUT_DIR}/opinions.csv", index=False)
    print("\nTop (aspect, opinion) pairs:\n",
          op.groupby(["aspect", "opinion"]).size().sort_values(ascending=False).head(15))
    cat = op[op.category != "other"].groupby("category")["polarity"].agg(["mean", "count"])
    print("\nAverage sentiment per aspect category:\n", cat)
    by = (op[op.category != "other"].groupby(["category", "label"])["polarity"]
          .mean().unstack().rename(columns={0: "negative reviews", 1: "positive reviews"}))
    print("\nAverage aspect sentiment split by review class:\n", by.round(3))
    by.plot(kind="bar", color=["tab:red", "tab:green"])
    plt.title("Aspect sentiment by review class"); plt.ylabel("mean VADER polarity")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/nlp_aspects_by_class.png", dpi=150); plt.close()
    cat["mean"].plot(kind="bar", color="tab:blue")
    plt.title("Opinion mining: average sentiment per aspect"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/nlp_aspects.png", dpi=150)

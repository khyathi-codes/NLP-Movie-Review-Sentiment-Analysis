# app.py  (Member 5)  FINAL NLP APPLICATION.   Run:  streamlit run app.py
import joblib
import pandas as pd
import streamlit as st
from config import MODEL_DIR
from preprocess import clean_text
from nlp_analysis import get_nlp, extract_opinions, WORD2ASPECT


@st.cache_resource
def load():
    return joblib.load(f"{MODEL_DIR}/final_model.joblib"), get_nlp()


model, nlp = load()
st.title("Movie Review Sentiment Analysis & Opinion Mining")
review = st.text_area("Paste a movie review", height=180,
                      value="The acting was brilliant and the music was wonderful, "
                            "but the plot was not interesting. Christopher Nolan "
                            "directed it.")

if st.button("Analyze") and review.strip():
    # 1. sentiment
    p = model.predict_proba([clean_text(review)])[0, 1]
    st.subheader("Sentiment: " + ("POSITIVE" if p >= 0.5 else "NEGATIVE"))
    st.progress(float(p))
    st.write(f"P(positive) = {p:.2%}")

    # 2. words that influenced the prediction
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    X = vec.transform([clean_text(review)])
    names = vec.get_feature_names_out()
    contrib = sorted(((names[i], X[0, i] * clf.coef_[0][i]) for i in X.nonzero()[1]),
                     key=lambda x: x[1])
    st.write("**Pushes towards negative:**", [w for w, c in contrib[:5] if c < 0])
    st.write("**Pushes towards positive:**", [w for w, c in contrib[::-1][:5] if c > 0])

    # 3. opinion mining + NER
    doc = nlp(review)
    ops = pd.DataFrame(extract_opinions(doc), columns=["aspect", "opinion", "polarity"])
    if not ops.empty:
        ops["category"] = ops["aspect"].map(lambda a: WORD2ASPECT.get(a, "other"))
        st.write("**Opinion mining (aspect -> opinion)**")
        st.dataframe(ops)
    ents = [(e.text, e.label_) for e in doc.ents]
    st.write("**Named entities:**", ents if ents else "none found")

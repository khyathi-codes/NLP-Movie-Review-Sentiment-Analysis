# NLP-Based Movie Review Sentiment Analysis and Opinion Mining System

Pipeline (as given by the faculty): Data collection -> Data analysis -> Text preprocessing ->
Vectorization (BoW + TF-IDF) / Word embedding (Word2Vec, CBOW) -> NLP analysis (POS, parsing, NER,
semantic labeling) -> Feature engineering -> ML classification -> Model comparison -> Evaluation -> Final application.

## Setup
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

## Run order (owner)
    python data_collection.py   # M1  -> data/reviews.csv   (skip: data/ already contains the team's files)
    python eda.py               # M1  -> outputs/eda_*.png
    python preprocess.py        # M2  -> data/clean.csv (clean_text + train/test split)
    python vectorize.py         # M2  -> BoW / TF-IDF matrices
    python embeddings.py        # M3  -> Word2Vec CBOW + Skip-gram, review vectors, PCA plots
    python features.py          # M3  -> 8 handcrafted features + correlation heatmap
    python nlp_analysis.py      # M4  -> POS, dependency parse, NER, SVO, aspect opinion mining
    python classify.py          # M5  -> outputs/model_comparison.csv/.png
    python evaluate.py          # M5  -> tuned final model, confusion matrix, ROC
    streamlit run app.py        # M5  -> final web application

`data/clean.csv` is the team's own M2 output (2000 reviews, 1600 train / 400 test). Do not change
RANDOM_STATE in config.py; all stages rely on the same split.

## Final results (400 held-out test reviews)
- Best model in the comparison: TF-IDF + Linear SVM, accuracy 0.8625, F1 0.8668
- Final app model (TF-IDF 1-2 grams + Logistic Regression, C tuned by 3-fold CV): accuracy 0.8675, ROC-AUC 0.9459
- Word2Vec (averaged CBOW vectors) is weaker (0.68-0.74) because the corpus is small (1600 training reviews)

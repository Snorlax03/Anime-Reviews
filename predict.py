"""Classify anime reviews as positive or negative with the trained model.

This applies the exact cleaning used in training (notebook 3, Step 8a) and the
frozen stop-word / entity lists in data/clean_artifacts.json, then runs the saved
Linear SVM pipeline. No scraping or training is needed.

Usage
-----
    python predict.py "The animation was stunning and the ending made me cry."
    python predict.py "review one" "review two"
    python predict.py --file reviews.txt        # one review per line

From Python
-----------
    from predict import predict
    predict(["Great show!", "Boring and rushed."])
    # -> [{'label': 'positive', 'score': 0.83, ...}, {'label': 'negative', ...}]

The score is the SVM's signed distance from the decision boundary: above 0 means
positive, below 0 negative, and the further from 0 the more confident the model.
Notebook 3 treats negative predictions with |score| < 0.5 as worth a human check.
"""
import argparse
import json
import os
import re
import sys

import joblib
import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "data", "final_sentiment_model.joblib")
ARTIFACTS_PATH = os.path.join(HERE, "data", "clean_artifacts.json")
LOW_CONFIDENCE = 0.5

for _pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(_pkg, quiet=True)

with open(ARTIFACTS_PATH, encoding="utf-8") as _f:
    _artifacts = json.load(_f)
STOPWORDS = set(_artifacts["stopwords"])
STRIP_SET = set(_artifacts["strip_set"])
META_COLS = _artifacts["keep_meta"]

_lemmatizer = WordNetLemmatizer()
HTML_TAG_RE, HTML_ENT_RE = re.compile(r"<[^>]+>"), re.compile(r"&[a-z]+;")
URL_RE, REPEAT_PUNC = re.compile(r"http\S+|www\.\S+"), re.compile(r"([!?.])\1+")
NON_ALPHA_RE, MULTISPACE = re.compile(r"[^a-z\s]"), re.compile(r"\s+")

_model = None


def _wn_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def clean_text(text):
    """Lowercase, strip markup and punctuation, POS-aware lemmatise, and remove
    stop-words and show/character names. Identical to the training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for rgx in (HTML_TAG_RE, HTML_ENT_RE, URL_RE):
        text = rgx.sub(" ", text)
    text = REPEAT_PUNC.sub(r"\1", text)
    text = MULTISPACE.sub(" ", NON_ALPHA_RE.sub(" ", text)).strip()
    out = []
    for tok, tag in pos_tag(word_tokenize(text)):
        if len(tok) <= 2 or tok in STOPWORDS:
            continue
        lemma = _lemmatizer.lemmatize(tok, _wn_pos(tag))
        if len(lemma) <= 2 or lemma in STOPWORDS or lemma in STRIP_SET:
            continue
        out.append(lemma)
    return " ".join(out)


def meta_features(text):
    """The stylistic features that survived screening in notebook 2, from RAW text."""
    if not isinstance(text, str) or not text:
        return {c: 0 for c in META_COLS}
    words = text.split()
    candidates = {
        "exclamation_count":    text.count("!"),
        "question_count":       text.count("?"),
        "uppercase_word_count": sum(1 for w in words if len(w) >= 3 and w.isupper()),
    }
    return {c: candidates.get(c, 0) for c in META_COLS}


def build_features(texts):
    """Turn raw review strings into the DataFrame the saved pipeline expects."""
    rows = [{"clean_text": clean_text(t), **meta_features(t)} for t in texts]
    return pd.DataFrame(rows, columns=["clean_text"] + META_COLS)


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(texts):
    """Classify a list of raw review strings. Returns one dict per review."""
    if isinstance(texts, str):
        texts = [texts]
    model = load_model()
    X = build_features(texts)
    labels = model.predict(X)
    scores = model.decision_function(X)
    return [{
        "label": "positive" if lab == 1 else "negative",
        "score": round(float(s), 3),
        "low_confidence": bool(abs(s) < LOW_CONFIDENCE),
        "text": t,
    } for t, lab, s in zip(texts, labels, scores)]


def main():
    parser = argparse.ArgumentParser(description="Classify anime reviews as positive or negative.")
    parser.add_argument("reviews", nargs="*", help="review text(s) to classify")
    parser.add_argument("--file", help="text file with one review per line")
    args = parser.parse_args()

    texts = list(args.reviews)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        parser.print_help()
        sys.exit(1)

    for r in predict(texts):
        flag = "  (low confidence)" if r["low_confidence"] else ""
        preview = " ".join(r["text"].split())[:70]
        print(f"{r['label']:<8} {r['score']:+.2f}{flag}  {preview}")


if __name__ == "__main__":
    main()

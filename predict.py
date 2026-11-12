"""Classify anime reviews as positive or negative with a trained model.

Three trained models can be used:

    svm         Linear SVM + TF-IDF (notebook 3). The default. Runs on a CPU, under 1 MB,
                included in the repo.
    distilbert  DistilBERT fine-tuned on the reviews (notebook 4).
    roberta     RoBERTa-base fine-tuned on the reviews (notebook 4).

The fine-tuned transformers are too large for GitHub (roughly 250-500 MB each). They are
loaded from models/<name>/ if you trained them with notebook 4, or downloaded from the
Hugging Face Hub if a repo id is set in HUB_MODEL_IDS below. They need `torch` and
`transformers`; a GPU is used automatically when one is available.

Usage
-----
    python predict.py "The animation was stunning and the ending made me cry."
    python predict.py "review one" "review two"
    python predict.py --file reviews.txt                 # one review per line
    python predict.py --model roberta "Great show!"      # use a fine-tuned transformer
    python predict.py --model-path path/or/hub-id "..."  # any saved fine-tuned model

From Python
-----------
    from predict import predict
    predict(["Great show!", "Boring and rushed."])                   # Linear SVM
    predict(["Great show!", "Boring and rushed."], model="roberta")  # fine-tuned RoBERTa

What the score means
--------------------
    svm          the SVM's signed distance from its decision boundary: above 0 is positive,
                 below 0 negative, and the further from 0 the more confident.
    transformers the probability that the review is positive (0 to 1).

A prediction is marked low_confidence when |score| < 0.5 for the SVM, or when the
probability is between 0.25 and 0.75 for a transformer. Notebook 4 recommends that a
person checks low-confidence negative predictions.
"""
import argparse
import json
import os
import re
import sys

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "data", "final_sentiment_model.joblib")
ARTIFACTS_PATH = os.path.join(HERE, "data", "clean_artifacts.json")
LOW_CONFIDENCE = 0.5

# ---- Fine-tuned transformers (notebook 4) ------------------------------------------
MODELS_DIR = os.path.join(HERE, "models")
TRANSFORMER_MODELS = {                      # short name -> folder name inside models/
    "distilbert": "distilbert-anime-sentiment",
    "roberta":    "roberta-base-anime-sentiment",
}
# Hugging Face Hub copies, used when models/<folder>/ doesn't exist on this machine.
# Notebook 4 (Section 12) can upload the models; then put the repo ids here, e.g.
#   "roberta": "your-hf-username/roberta-base-anime-sentiment"
HUB_MODEL_IDS = {
    "distilbert": None,
    "roberta":    None,
}
MAX_LENGTH = 512      # both models read at most 512 tokens
HEAD_TOKENS = 128     # long reviews keep their first 128 and last 382 tokens (see notebook 4)

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
_transformers = {}          # cache of loaded (model, tokenizer, info), keyed by source + device


# =====================================================================================
# Linear SVM (notebook 3)
# =====================================================================================
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
    if isinstance(texts, str):
        texts = [texts]
    rows = [{"clean_text": clean_text(t), **meta_features(t)} for t in texts]
    return pd.DataFrame(rows, columns=["clean_text"] + META_COLS)


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_svm(texts):
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


# =====================================================================================
# Fine-tuned transformers (notebook 4)
# =====================================================================================
def encode_head_tail(tokenizer, texts, max_length=MAX_LENGTH, head_tokens=HEAD_TOKENS):
    """Token ids for each review, special tokens included.

    A review longer than `max_length` tokens keeps its first `head_tokens` tokens and its
    last `max_length - head_tokens - 2` tokens: the opening plus the closing verdict,
    which is where reviewers tend to sum up. Notebook 4 trains with this function and
    prediction uses it too, so the model always sees reviews cut the same way.
    `head_tokens = max_length - 2` gives plain head truncation (what the zero-shot
    pipelines in notebook 3 did)."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [t if isinstance(t, str) else "" for t in texts]
    tail_tokens = max_length - head_tokens - 2
    if tail_tokens < 0:
        raise ValueError("head_tokens must be at most max_length - 2")
    ids = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    out = []
    for seq in ids:
        if len(seq) > max_length:
            # seq = [CLS/<s>] tokens... [SEP/</s>]; keep the first and last special token
            seq = seq[:1 + head_tokens] + seq[len(seq) - tail_tokens - 1:]
        out.append(seq)
    return out


def pad_batch(id_lists, pad_id, device=None):
    """Right-pad a list of token-id lists into input_ids / attention_mask tensors."""
    import torch
    width = max(len(x) for x in id_lists)
    input_ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for i, seq in enumerate(id_lists):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = 1
    if device is not None:
        input_ids, mask = input_ids.to(device), mask.to(device)
    return {"input_ids": input_ids, "attention_mask": mask}


def positive_index(model):
    """Which output column means 'positive' (works for our models and the Hub ones)."""
    for idx, name in model.config.id2label.items():
        if str(name).lower().startswith("pos"):
            return int(idx)
    return 1


def predict_proba_ids(model, id_lists, pad_id, device, batch_size=32, fp16=True):
    """P(positive) for already-encoded reviews. Batches are formed from reviews of
    similar length, so little compute is wasted on padding."""
    import torch
    pos = positive_index(model)
    order = np.argsort([len(x) for x in id_lists], kind="stable")
    probs = np.empty(len(id_lists), dtype=np.float64)
    use_amp = fp16 and torch.device(device).type == "cuda"
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = pad_batch([id_lists[i] for i in idx], pad_id, device)
            with torch.autocast(device_type=torch.device(device).type,
                                dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1)[:, pos].cpu().numpy()
    if was_training:
        model.train()
    return probs


def transformer_proba(model, tokenizer, texts, device, batch_size=32, fp16=True,
                      max_length=MAX_LENGTH, head_tokens=HEAD_TOKENS):
    """P(positive) for raw review strings."""
    ids = encode_head_tail(tokenizer, texts, max_length, head_tokens)
    return predict_proba_ids(model, ids, tokenizer.pad_token_id, device, batch_size, fp16)


def resolve_model_source(name, models_dir=None, allow_hub=True):
    """Where to load a fine-tuned model from: a local folder, a Hub repo id, or None."""
    models_dir = models_dir or MODELS_DIR
    if name in TRANSFORMER_MODELS:
        local = os.path.join(models_dir, TRANSFORMER_MODELS[name])
        if os.path.exists(os.path.join(local, "config.json")):
            return local
        return HUB_MODEL_IDS.get(name) if allow_hub else None
    return name           # already a folder path or a Hub repo id


def load_training_info(source):
    """training_info.json saved next to a fine-tuned model (local or on the Hub)."""
    local = os.path.join(source, "training_info.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    if not os.path.isdir(source):
        try:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(source, "training_info.json"), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer(name_or_path, device=None):
    """Load (model, tokenizer, info) for 'distilbert', 'roberta', a folder or a Hub id."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()      # silences "sequence longer than 512" notices

    source = resolve_model_source(name_or_path)
    if source is None:
        raise FileNotFoundError(
            f"No fine-tuned '{name_or_path}' model found. Train it with "
            f"04_Transformer_Finetuning.ipynb (it is saved to models/"
            f"{TRANSFORMER_MODELS[name_or_path]}/), or set HUB_MODEL_IDS['{name_or_path}'] "
            f"in predict.py to a Hugging Face Hub repo id.")
    device = torch.device(device) if device is not None else _default_device()
    key = (source, str(device))
    if key not in _transformers:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source).to(device).eval()
        _transformers[key] = (model, tokenizer, load_training_info(source))
    return _transformers[key]


def _predict_transformer(texts, name_or_path, device=None):
    model, tokenizer, info = load_transformer(name_or_path, device)
    probs = transformer_proba(model, tokenizer, texts, model.device,
                              max_length=info.get("max_length", MAX_LENGTH),
                              head_tokens=info.get("head_tokens", HEAD_TOKENS))
    return [{
        "label": "positive" if p >= 0.5 else "negative",
        "score": round(float(p), 3),
        "low_confidence": bool(abs(2 * p - 1) < LOW_CONFIDENCE),
        "text": t,
    } for t, p in zip(texts, probs)]


# =====================================================================================
# Public API
# =====================================================================================
def predict(texts, model="svm", device=None):
    """Classify a list of raw review strings. Returns one dict per review.

    model: "svm" (default), "distilbert", "roberta", or a folder / Hub id of a model
    fine-tuned by notebook 4. device: e.g. "cpu" or "cuda" (transformers only)."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if model == "svm":
        return _predict_svm(texts)
    return _predict_transformer(texts, model, device)


def main():
    parser = argparse.ArgumentParser(description="Classify anime reviews as positive or negative.")
    parser.add_argument("reviews", nargs="*", help="review text(s) to classify")
    parser.add_argument("--file", help="text file with one review per line")
    parser.add_argument("--model", default="svm", choices=["svm", *TRANSFORMER_MODELS],
                        help="which trained model to use (default: svm)")
    parser.add_argument("--model-path", help="folder or Hugging Face Hub id of a fine-tuned model "
                                             "(overrides --model)")
    parser.add_argument("--cpu", action="store_true", help="run transformers on the CPU even if a GPU exists")
    args = parser.parse_args()

    texts = list(args.reviews)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        parser.print_help()
        sys.exit(1)

    model = args.model_path or args.model
    for r in predict(texts, model=model, device="cpu" if args.cpu else None):
        flag = "  (low confidence)" if r["low_confidence"] else ""
        score = f"{r['score']:+.2f}" if model == "svm" else f"{r['score']:.2f}"
        preview = " ".join(r["text"].split())[:70]
        print(f"{r['label']:<8} {score}{flag}  {preview}")


if __name__ == "__main__":
    main()

"""
Roman Urdu Sentiment Analysis — Full Pipeline
=============================================
Dataset: Roman_Urdu_DataSet.csv
Classes: Positive, Negative, Neutral
Best Model: Logistic Regression (word + char n-gram TF-IDF)
Accuracy: ~66%  |  Weighted F1: ~0.659

Key techniques applied:
  - Spelling normalization dictionary
  - Negation scope tagging (NEG_*)
  - Intensifier marking (INTENS_*)
  - Emoji → sentiment tokens
  - Word n-grams (1-3) + Char n-grams (2-5)
  - Class-balanced training weights
"""

import re
import warnings
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════

DATA_PATH = 'Roman Urdu DataSet.csv'   # <- change if needed
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════
# 2. NORMALIZATION DICTIONARIES
# ══════════════════════════════════════════════════════════════

NORM_DICT = {
    # negation
    "nhi": "nahi", "nhe": "nahi", "ni": "nahi", "nh": "nahi",
    # common words
    "bht": "bohat", "bhut": "bohat", "bouth": "bohat",
    "bohot": "bohat", "boht": "bohat",
    "hy": "hai", "hain": "hai", "he": "hai", "ha": "hai",
    "kr": "kar", "kro": "karo", "krna": "karna", "krdo": "kar_do",
    "acha": "acha", "achha": "acha", "accha": "acha",
    "thk": "theek", "thek": "theek", "thik": "theek",
    "phr": "phir", "sb": "sab", "ap": "aap", "yr": "yaar",
    "hm": "hum", "tm": "tum", "mn": "main", "mein": "main",
    "or": "aur", "lkn": "lekin", "agr": "agar",
    "hogya": "ho_gaya", "hogaya": "ho_gaya",
    "lga": "laga", "lgta": "lagta", "lgti": "lagti",
    "kuch": "kuch", "kch": "kuch",
    "bhi": "bhi", "bi": "bhi",
    "pta": "pata", "smjh": "samajh",
    "allah": "allah", "alah": "allah",
    "mujhe": "mujhe", "mjhe": "mujhe", "muje": "mujhe",
    "koi": "koi", "koy": "koi",
    "zindagi": "zindagi", "zndgi": "zindagi",
    "mashallah": "mashallah", "masha_allah": "mashallah",
    "alhamdulillah": "alhamdulillah", "alhmdulilaah": "alhamdulillah",
}

NEGATION_WORDS = {
    "nahi", "nhi", "mat", "na", "no", "not",
    "never", "bilkul_nahi", "kabhi_nahi", "hargiz"
}

INTENSIFIERS = {
    "bohat", "bahut", "zyada", "bilkul",
    "ekdum", "zabardast", "kamaal"
}

# ══════════════════════════════════════════════════════════════
# 3. PREPROCESSING FUNCTION
# ══════════════════════════════════════════════════════════════

def preprocess(texts):
    """
    Full Roman Urdu preprocessing pipeline.
    Accepts a list of strings, returns a list of cleaned strings.
    """
    out = []
    for text in texts:
        text = str(text).lower().strip()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)

        # Map emojis to sentiment tokens (before removing punctuation)
        text = re.sub(r'😊|😄|😁|😀|🙂', ' EMOJI_HAPPY ', text)
        text = re.sub(r'😢|😭|😞|😔|💔', ' EMOJI_SAD ', text)
        text = re.sub(r'😂|😜|😝|😋', ' EMOJI_LAUGH ', text)
        text = re.sub(r'😡|😠|👿', ' EMOJI_ANGRY ', text)
        text = re.sub(r'❤|💕|💗|💓', ' EMOJI_LOVE ', text)

        # Preserve sentiment punctuation as tokens
        text = re.sub(r'!+', ' EXCLAIM ', text)
        text = re.sub(r'\?+', ' QUESTION ', text)

        # Remove remaining punctuation
        text = re.sub(r'[^\w\s]', ' ', text)

        # Collapse repeated characters: "achaaa" → "achaa" (max 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize spelling variants
        tokens = text.split()
        tokens = [NORM_DICT.get(t, t) for t in tokens]

        # Negation scope: tag NEG_ after negation words
        result, negate = [], False
        for t in tokens:
            if t in NEGATION_WORDS:
                negate = True
                result.append(t)
            elif t in {'EXCLAIM', 'QUESTION'}:
                negate = False
                result.append(t)
            elif negate:
                result.append('NEG_' + t)
            elif t in INTENSIFIERS:
                result.append('INTENS_' + t)
            else:
                result.append(t)

        out.append(' '.join(result))
    return out


# ══════════════════════════════════════════════════════════════
# 4. LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_data(path):
    df = pd.read_csv(path, header=None, names=['text', 'label', 'extra'])
    df = df[['text', 'label']].dropna(subset=['text'])
    df['label'] = df['label'].str.strip()
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])].reset_index(drop=True)
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 1].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# 5. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def build_features(X_train_proc, X_test_proc):
    """
    Combine word n-gram TF-IDF and char n-gram TF-IDF.
    Returns sparse matrices and fitted vectorizers.
    """
    word_v = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )
    char_v = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        max_features=30000,
        sublinear_tf=True,
        min_df=3,
    )
    Xw_train = word_v.fit_transform(X_train_proc)
    Xw_test  = word_v.transform(X_test_proc)
    Xc_train = char_v.fit_transform(X_train_proc)
    Xc_test  = char_v.transform(X_test_proc)

    X_train_f = sp.hstack([Xw_train, Xc_train])
    X_test_f  = sp.hstack([Xw_test,  Xc_test])
    return X_train_f, X_test_f, word_v, char_v


# ══════════════════════════════════════════════════════════════
# 6. TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════

def train_all(X_train_f, X_test_f, y_train, y_test, cwd):
    results = {}

    # Logistic Regression
    lr = LogisticRegression(C=5.0, max_iter=1000, solver='saga',
                             class_weight=cwd, n_jobs=-1)
    lr.fit(X_train_f, y_train)
    p = lr.predict(X_test_f)
    results['LogisticRegression'] = {
        'acc': accuracy_score(y_test, p),
        'f1':  f1_score(y_test, p, average='weighted'),
        'pred': p, 'model': lr
    }

    # LinearSVC
    svc = LinearSVC(C=1.0, max_iter=2000, class_weight=cwd)
    svc.fit(X_train_f, y_train)
    p = svc.predict(X_test_f)
    results['LinearSVC'] = {
        'acc': accuracy_score(y_test, p),
        'f1':  f1_score(y_test, p, average='weighted'),
        'pred': p, 'model': svc
    }

    # ComplementNB (needs non-negative features)
    scaler = MaxAbsScaler()
    Xs_train = scaler.fit_transform(X_train_f)
    Xs_test  = scaler.transform(X_test_f)
    Xs_train.data = np.abs(Xs_train.data)
    Xs_test.data  = np.abs(Xs_test.data)
    cnb = ComplementNB(alpha=0.3)
    cnb.fit(Xs_train, y_train)
    p = cnb.predict(Xs_test)
    results['ComplementNB'] = {
        'acc': accuracy_score(y_test, p),
        'f1':  f1_score(y_test, p, average='weighted'),
        'pred': p, 'model': cnb, 'scaler': scaler
    }

    return results


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  ROMAN URDU SENTIMENT ANALYSIS")
    print("=" * 60)

    # Load
    df = load_data(DATA_PATH)
    print(f"\nDataset: {len(df):,} samples")
    print(df['label'].value_counts().to_string())

    # Preprocess
    print("\nPreprocessing text...")
    X_proc = preprocess(df['text'].values)
    y = df['label'].values

    # Split
    X_train_p, X_test_p, y_train, y_test = train_test_split(
        X_proc, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train_p):,}  |  Test: {len(X_test_p):,}")

    # Class weights
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    cwd = dict(zip(classes, cw))

    # Features
    print("Extracting features (word + char n-grams)...")
    X_train_f, X_test_f, word_v, char_v = build_features(X_train_p, X_test_p)
    print(f"Feature matrix: {X_train_f.shape}")

    # Train
    print("\nTraining models...")
    results = train_all(X_train_f, X_test_f, y_train, y_test, cwd)

    # Report
    print("\n── Model Comparison ──")
    for name, r in results.items():
        print(f"  {name:25s}  Accuracy={r['acc']:.4f}  F1={r['f1']:.4f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, best['pred'],
        target_names=['Negative', 'Neutral', 'Positive']
    ))

    # ── Inference function ──────────────────────────────────
    def predict(texts):
        """
        Predict sentiment for a list of Roman Urdu strings.
        Returns list of labels: Positive / Negative / Neutral
        """
        proc = preprocess(texts)
        Dw = word_v.transform(proc)
        Dc = char_v.transform(proc)
        Df = sp.hstack([Dw, Dc])
        return best['model'].predict(Df)

    # Demo
    demo_texts = [
        "Bohat acha kaam kia hai, zabardast!",
        "Yeh bilkul bhi acha nahi tha, bura laga",
        "Pata nahi kya hoga kal",
        "Mashallah bohat khubsoorat jagah hai",
        "Nahi pasand aaya mujhe bilkul",
        "Thik hai koi baat nahi",
    ]

    preds = predict(demo_texts)
    print("\n── Demo Predictions ──")
    for text, pred in zip(demo_texts, preds):
        print(f"  [{pred:8s}]  {text}")

    print(f"\nFinal Accuracy : {best['acc']*100:.1f}%")
    print(f"Final F1 Score : {best['f1']:.4f}")

#save model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best['model'],
        'word_vectorizer': word_v,
        'char_vectorizer': char_v,
        'class_weights': cwd
    }, f)
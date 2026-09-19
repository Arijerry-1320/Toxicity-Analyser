"""
train_model.py - Toxicity Comment Classification
=================================================
Dataset  : data/data.csv  (Jigsaw / Civil Comments toxicity dataset)
Target   : binary 'toxic' label  (1 if target >= 0.5, else 0)
Features : TF-IDF on comment_text  +  structured sub-scores
           (severe_toxicity, obscene, identity_attack, insult, threat,
            sexual_explicit, funny, wow, sad, likes, disagree,
            identity_annotator_count, toxicity_annotator_count)
Models   : Logistic Regression, Linear SVC, Random Forest, Gradient Boosting
Best     : saved to models/model.pkl along with metadata
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)

from pipeline_utils import ColumnSelector, TextSelector

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
DATA_PATH  = os.path.join("data", "data.csv")
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# -- Feature columns (always present, no NaN) --------------------------------
STRUCT_FEATURES = [
    "severe_toxicity", "obscene", "identity_attack",
    "insult", "threat", "sexual_explicit",
    "funny", "wow", "sad", "likes", "disagree",
    "identity_annotator_count", "toxicity_annotator_count",
]
TEXT_FEATURE = "comment_text"
TARGET_COL   = "target"
THRESHOLD    = 0.5   # target >= THRESHOLD -> toxic = 1


# -- Load & prepare data -----------------------------------------------------
def load_data():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    # Binary label
    df["toxic"] = (df[TARGET_COL] >= THRESHOLD).astype(int)

    # Fill any remaining NaN in structured features with 0
    df[STRUCT_FEATURES] = df[STRUCT_FEATURES].fillna(0)
    df[TEXT_FEATURE]    = df[TEXT_FEATURE].fillna("")

    print(f"  Class distribution - toxic=1: {df['toxic'].sum():,} "
          f"({df['toxic'].mean()*100:.1f}%)  "
          f"toxic=0: {(1 - df['toxic']).sum():,} "
          f"({(1 - df['toxic'].mean())*100:.1f}%)")
    return df


# -- Build feature pipeline --------------------------------------------------
def build_pipeline(classifier):
    text_pipeline = Pipeline([
        ("selector", TextSelector(TEXT_FEATURE)),
        ("tfidf",    TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            strip_accents="unicode",
        )),
    ])

    struct_pipeline = Pipeline([
        ("selector", ColumnSelector(STRUCT_FEATURES)),
        ("scaler",   StandardScaler()),
    ])

    features = FeatureUnion([
        ("text",   text_pipeline),
        ("struct", struct_pipeline),
    ])

    return Pipeline([
        ("features",   features),
        ("classifier", classifier),
    ])


# -- Train & evaluate all candidates -----------------------------------------
def train_and_compare(X_train, X_test, y_train, y_test):
    candidates = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", n_jobs=-1
        ),
        "Linear SVC": LinearSVC(
            C=0.5, max_iter=2000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42,
            subsample=0.8
        ),
    }

    results = {}
    best_name, best_f1, best_pipeline = None, -1, None

    for name, clf in candidates.items():
        print(f"\n--- Training: {name} ---")
        pipeline = build_pipeline(clf)

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)

        # LinearSVC has no predict_proba; fall back to decision_function for AUC
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except AttributeError:
            y_prob = pipeline.decision_function(X_test)
            auc = roc_auc_score(y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="binary")

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Time     : {train_time:.1f}s")
        print(classification_report(y_test, y_pred,
                                    target_names=["Non-Toxic", "Toxic"]))

        results[name] = {
            "accuracy": round(acc, 4),
            "f1":       round(f1, 4),
            "roc_auc":  round(auc, 4),
            "train_time_s": round(train_time, 2),
        }

        if f1 > best_f1:
            best_f1       = f1
            best_name     = name
            best_pipeline = pipeline

    return best_name, best_pipeline, results


# -- Main --------------------------------------------------------------------
def main():
    df = load_data()

    X = df[[TEXT_FEATURE] + STRUCT_FEATURES]
    y = df["toxic"]

    print("\nSplitting data (80/20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    best_name, best_pipeline, results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    print(f"\n{'*'*60}")
    print(f"  Best model : {best_name}")
    print(f"  F1-score   : {results[best_name]['f1']:.4f}")
    print(f"  ROC-AUC    : {results[best_name]['roc_auc']:.4f}")
    print(f"{'*'*60}\n")

    # Save model
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH}")

    # Save metadata (used by app.py)
    meta = {
        "best_model": best_name,
        "target_column": TARGET_COL,
        "binary_label": "toxic",
        "threshold": THRESHOLD,
        "text_feature": TEXT_FEATURE,
        "structured_features": STRUCT_FEATURES,
        "task": "binary_classification",
        "label_map": {"0": "Non-Toxic", "1": "Toxic"},
        "model_results": results,
        "test_metrics": results[best_name],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata  -> {META_PATH}")

    # Quick smoke test: reload and predict one sample
    print("\nSmoke test - reloading model ...")
    loaded = joblib.load(MODEL_PATH)
    sample = X_test.iloc[[0]]
    pred   = loaded.predict(sample)[0]
    actual = y_test.iloc[0]
    label  = "Toxic" if pred == 1 else "Non-Toxic"
    print(f"  Predicted: {pred} ({label})  Actual: {actual}  OK")
    print("\nDone.")


if __name__ == "__main__":
    main()

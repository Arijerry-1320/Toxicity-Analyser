"""
app.py - Toxicity Comment Classifier (Streamlit)
=================================================
Load the trained pipeline from models/model.pkl and let users enter a
comment + optional structured scores to receive a real-time toxicity
prediction.

Run locally:
    streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import custom transformers so joblib can deserialise the pipeline
from pipeline_utils import ColumnSelector, TextSelector  # noqa: F401

# -- Paths -------------------------------------------------------------------
MODEL_PATH = "model.pkl"
META_PATH  = "model_meta.json"

# -- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Toxicity Comment Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Load model & metadata (cached) -----------------------------------------
@st.cache_resource(show_spinner="Loading ML model ...")
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta


def run():
    # ---- Header -----------------------------------------------------------
    st.title("🔍 Comment Toxicity Classifier")
    st.markdown(
        """
        This app uses a machine-learning pipeline (TF-IDF + structured features)
        trained on the **Civil Comments / Jigsaw toxicity dataset** to predict
        whether a comment is **Toxic** or **Non-Toxic**.
        """
    )

    # ---- Load model -------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`.  "
            "Please run `python train_model.py` first."
        )
        st.stop()

    pipeline, meta = load_model()
    best_model = meta.get("best_model", "Unknown")
    struct_features = meta["structured_features"]
    text_feature    = meta["text_feature"]

    # ---- Sidebar: model info ----------------------------------------------
    with st.sidebar:
        st.header("Model Info")
        st.markdown(f"**Best model:** {best_model}")
        test_metrics = meta.get("test_metrics", {})
        if test_metrics:
            st.metric("Accuracy",  f"{test_metrics.get('accuracy', 0):.4f}")
            st.metric("F1-Score",  f"{test_metrics.get('f1', 0):.4f}")
            st.metric("ROC-AUC",   f"{test_metrics.get('roc_auc', 0):.4f}")

        st.divider()
        st.markdown(
            "**Task:** Binary Classification  \n"
            "**Threshold:** target ≥ 0.5 → Toxic  \n"
            "**Text feature:** TF-IDF (unigrams + bigrams)  \n"
            f"**Structured features:** {len(struct_features)}"
        )

        st.divider()
        st.markdown("### All Model Results")
        model_results = meta.get("model_results", {})
        if model_results:
            rows = []
            for name, m in model_results.items():
                rows.append({
                    "Model": name,
                    "Accuracy": m["accuracy"],
                    "F1": m["f1"],
                    "AUC": m["roc_auc"],
                    "Time(s)": m["train_time_s"],
                })
            st.dataframe(
                pd.DataFrame(rows).set_index("Model"),
                use_container_width=True,
            )

    # ---- Main input area --------------------------------------------------
    st.subheader("Enter a Comment")
    comment = st.text_area(
        "Comment text",
        placeholder="Type or paste a comment here ...",
        height=140,
        help="The comment text is vectorised with TF-IDF (20 000 features, bigrams).",
    )

    st.subheader("Optional: Structured Sub-Scores")
    st.markdown(
        "These are crowd-sourced annotation scores (0.0 – 1.0). "
        "If you don't have them, leave all sliders at **0.0**."
    )

    # Two-column layout for sliders
    col1, col2 = st.columns(2)
    slider_vals = {}

    slider_labels = {
        "severe_toxicity":             "Severe Toxicity",
        "obscene":                     "Obscene",
        "identity_attack":             "Identity Attack",
        "insult":                      "Insult",
        "threat":                      "Threat",
        "sexual_explicit":             "Sexually Explicit",
        "funny":                       "Funny (reaction count)",
        "wow":                         "Wow (reaction count)",
        "sad":                         "Sad (reaction count)",
        "likes":                       "Likes (reaction count)",
        "disagree":                    "Disagree (reaction count)",
        "identity_annotator_count":    "Identity Annotator Count",
        "toxicity_annotator_count":    "Toxicity Annotator Count",
    }

    # The reaction/count features can exceed 1.0, so give them a bigger range
    count_features = {"funny", "wow", "sad", "likes", "disagree",
                      "identity_annotator_count", "toxicity_annotator_count"}

    for i, feat in enumerate(struct_features):
        col = col1 if i % 2 == 0 else col2
        label = slider_labels.get(feat, feat.replace("_", " ").title())
        if feat in count_features:
            slider_vals[feat] = col.number_input(
                label, min_value=0, max_value=500, value=0, step=1
            )
        else:
            slider_vals[feat] = col.slider(
                label, min_value=0.0, max_value=1.0, value=0.0, step=0.01
            )

    # ---- Predict ----------------------------------------------------------
    st.divider()
    predict_btn = st.button("Predict Toxicity", type="primary", use_container_width=True)

    if predict_btn:
        if not comment.strip():
            st.warning("Please enter some comment text before predicting.")
            st.stop()

        # Build input DataFrame matching training schema
        input_data = {text_feature: [comment.strip()]}
        for feat in struct_features:
            input_data[feat] = [float(slider_vals[feat])]

        input_df = pd.DataFrame(input_data)

        # Run prediction
        prediction = pipeline.predict(input_df)[0]

        # Confidence / probability (not all estimators support it)
        try:
            proba = pipeline.predict_proba(input_df)[0]
            toxic_prob     = proba[1]
            non_toxic_prob = proba[0]
            has_proba = True
        except AttributeError:
            decision = pipeline.decision_function(input_df)[0]
            toxic_prob = float(1 / (1 + np.exp(-decision)))   # sigmoid
            non_toxic_prob = 1 - toxic_prob
            has_proba = False

        # ---- Display result -----------------------------------------------
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨  **TOXIC**  — This comment appears to be toxic.")
        else:
            st.success("✅  **NON-TOXIC**  — This comment appears to be non-toxic.")

        # Confidence bar
        r1, r2 = st.columns(2)
        r1.metric("Toxic probability",     f"{toxic_prob*100:.1f}%")
        r2.metric("Non-toxic probability", f"{non_toxic_prob*100:.1f}%")

        st.progress(float(toxic_prob), text=f"Toxicity confidence: {toxic_prob*100:.1f}%")

        if not has_proba:
            st.caption(
                "_Note: probability is approximated from the decision function "
                "since this model does not natively output probabilities._"
            )

        # ---- Raw input summary -------------------------------------------
        with st.expander("Show input sent to model"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    run()

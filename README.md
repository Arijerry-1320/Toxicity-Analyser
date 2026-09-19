# Toxicity Comment Classifier

A machine-learning web application that predicts whether an online comment is **Toxic** or **Non-Toxic**, built with **scikit-learn** and **Streamlit**.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `data/data.csv` |
| Source | Civil Comments / Jigsaw Unintended Bias in Toxicity Classification |
| Rows | 90,902 |
| Target | `target` (continuous 0–1) binarised at ≥ 0.5 |
| Task | Binary Classification |

---

## Model Pipeline

1. **Text branch** — TF-IDF (20 000 features, unigrams + bigrams) on `comment_text`
2. **Structured branch** — StandardScaler on 13 numeric sub-scores
3. **Feature union** of both branches
4. **Classifier** — best of four candidates (Logistic Regression, Linear SVC, Random Forest, Gradient Boosting) selected by F1-score

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.9991 | 0.9991 | 1.0000 |
| Linear SVC | 0.9993 | 0.9993 | 1.0000 |
| Random Forest | 0.9114 | 0.9054 | 0.9896 |
| **Gradient Boosting** *(best)* | **0.9997** | **0.9997** | **1.0000** |

---

## Project Structure

```
project/
├── app.py                  # Streamlit web app
├── train_model.py          # Training script
├── requirements.txt        # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── data.csv            # Original dataset
└── models/
    ├── model.pkl           # Trained pipeline (generated)
    └── model_meta.json     # Metadata & metrics (generated)
```

---

## Local Setup & Run

### 1. Clone / open the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This generates `models/model.pkl` and `models/model_meta.json`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

---

## Streamlit Community Cloud Deployment

1. Push the project to a **public GitHub repository** (include `data/`, `models/`, all `.py` files, and `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs dependencies automatically.

> **Note:** The `models/model.pkl` file must be committed to the repository, or the deployment startup command must run `python train_model.py` before `streamlit run app.py`.  
> For large model files, consider using [Git LFS](https://git-lfs.github.com/).

---

## Environment Variables

No API keys are required. If you extend the app with external services, copy `.env.example` to `.env` and fill in the required values.

---

## Usage

1. Type or paste any comment into the text area.
2. *(Optional)* Adjust the structured sub-score sliders if you have annotation data.
3. Click **Predict Toxicity**.
4. The app returns **Toxic / Non-Toxic** with a confidence percentage.

# Aspect-Based Sentiment Analysis (ABSA)

**Academic Sem-5 | Module V: Natural Language Processing**  
**Model Architecture:** PyABSA with DeBERTa-v3  
**Benchmark Dataset:** SemEval-2014 Task 4  
**Demonstration Dataset:** Amazon Reviews Dataset  

---

## 1. Project Overview

Standard sentiment analysis labels an entire sentence as simply "positive" or "negative". However, real customer reviews frequently express contrasting feelings about different features in the same sentence:

> *"The **screen** is gorgeous, but the **battery life** is terrible."*

**Aspect-Based Sentiment Analysis (ABSA)** solves this problem in two simple steps:
1. **Find the Features (Aspect Extraction):** Identify what specific things are being reviewed (e.g., `screen`, `battery life`).
2. **Classify the Sentiment (Polarity Classification):** Determine whether the opinion on each individual feature is **Positive**, **Negative**, or **Neutral**.

---

## 2. How the Pipeline Works

```text
[ Customer Review Text ]
          │
          ▼
[ Language Model (DeBERTa-v3) ]
          │
          ├───► [ 1. Find Product Features (e.g. Battery, Screen) ]
          │              │
          │              ▼
          └───► [ 2. Zoom in on Nearby Words (Local Context Focus) ]
                         │
                         ▼
                [ 3. Classify Sentiment ]
             (Positive, Negative, Neutral)
```

1. **Sentence Reading (DeBERTa-v3):** The model reads the full sentence at once, understanding the relationship between words based on context.
2. **Feature Extraction:** Scans and highlights individual product or service features.
3. **Local Context Focus:** Zooms in on the words immediately around each feature. When checking `battery`, it focuses on "terrible" and ignores "gorgeous", preventing sentiment mix-ups.
4. **Sentiment Classification:** Assigns a sentiment label and confidence percentage to each feature.

---

## 3. Repository Structure

```text
nlp_lab_project/
├── app.py                     # Interactive Streamlit Web Dashboard
├── nlp_engine.py              # Core PyABSA Inference Engine
├── clean_data.py              # Dataset preparation and sampling script
├── eval_semeval.py            # SemEval-2014 Academic Evaluation script
├── eval_semeval2014.ipynb     # Academic Evaluation Jupyter Notebook
├── evaluation_results.json    # Serialized academic metrics report
├── requirements.txt           # Dependency specification
├── run_app.bat                # Single-click launcher
├── data/
│   ├── sample_amazon_reviews.csv   # Cleaned 500-review sample
│   └── semeval2014/                # Canonical SemEval-2014 Task 4 files
└── checkpoints/
    └── ATEPC_ENGLISH_CHECKPOINT/   # Local DeBERTa PyABSA model weights
```

---

## 4. How to Run

### 1. Setup Environment
```powershell
uv venv .venv --python 3.11
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
```powershell
python clean_data.py
```

### 3. Launch Dashboard
```powershell
streamlit run app.py
```
*(Or double-click `run_app.bat`)*

### 4. Run Academic Evaluation
```powershell
python eval_semeval.py
```

---

## 5. Benchmark Results: SemEval-2014 Task 4

| Task | Metric | Older Baseline (BiLSTM) | Standard BERT | **Our Model (DeBERTa-v3)** |
|---|---|:---:|:---:|:---:|
| **Finding Features (Extraction)** | Precision | 69.1% | 79.5% | **91.45%** |
| **Finding Features (Extraction)** | Recall | 67.8% | 78.9% | **96.40%** |
| **Finding Features (Extraction)** | **F1-Score** | 68.4% | 79.2% | **93.86%** |
| **Classifying Sentiment** | **Overall Accuracy** | 71.2% | 81.5% | **98.13%** |
| **Classifying Sentiment** | **Macro F1-Score** | 66.5% | 78.3% | **97.14%** |
| **Classifying Sentiment** | **Weighted F1-Score** | 70.8% | 81.1% | **98.12%** |

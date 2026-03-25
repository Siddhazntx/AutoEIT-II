# 🚀 AutoEIT: Automated Scoring for Spanish Elicited Imitation Tasks
**Specific Test II Submission**

AutoEIT is an end-to-end, reproducible NLP pipeline for automatically scoring Spanish **Elicited Imitation Task (EIT)** responses against target prompt sentences using a meaning-based rubric. Rather than depending purely on black-box LLMs, the system combines a deterministic **Early Quality Gate**, linguistically informed feature extraction, and an interpretable **Heuristic Mathematical Scorer**, with a lightweight ordinal ML baseline for comparison.

The project is designed to be:

- **Interpretable** — feature contributions and thresholds are transparent
- **Efficient** — obvious cases are resolved before expensive semantic inference
- **Reproducible** — configuration-driven, modular, and research-friendly
- **Low-resource aware** — built to perform well even with limited labeled data

---

## 🏆 Performance Summary

The AutoEIT pipeline predicts scores on a **0–4 rubric scale**.
Because the dataset is highly imbalanced (with a large majority of responses receiving score 4), model selection and evaluation are driven primarily by **Quadratic Weighted Kappa (QWK)** on a stratified validation split.

### A/B Evaluation Results

| Metric | Approach A (Heuristic Scorer) | Approach B (Ordinal ML) |
| :--- | :---: | :---: |
| **Quadratic Weighted Kappa (QWK)** | **0.8187** | 0.7956 |
| **Accuracy** | **76.92%** | 69.55% |
| **Macro F1-Score** | **0.5854** | 0.4901 |

**Conclusion:** The interpretable heuristic approach achieved the strongest validation performance, reaching **expert-level agreement** with human scoring while remaining fully transparent and easy to audit.

---

## 🧠 System Architecture

The pipeline follows a staged architecture designed for both **efficiency** and **rubric alignment**.

### 1. Text Cleaning & Normalization
Input transcriptions are standardized to reduce noise before scoring.

This includes:
- lowercasing
- punctuation normalization
- accent normalization
- removal of transcription artifacts such as pauses, cough markers, and gibberish tags

---

### 2. Early Quality Gate
A lightweight deterministic filter handles trivial cases before any expensive NLP inference.

Rules:
- **Exact match with target stimulus** → assign **Score 4**
- **Empty / pure gibberish response** → assign **Score 0**
- otherwise → continue to feature extraction

This optimization reduces unnecessary compute and isolates the ambiguous middle-range responses that actually require modeling.

---

### 3. Linguistic Feature Extraction
For unresolved cases, the pipeline extracts linguistically meaningful features using **spaCy**.

Features include:
- **Lemma Recall** — how much core lexical content from the target was retained
- **Idea-Unit / Content-Word Recall** — overlap over meaning-bearing words only

These features provide interpretable evidence for partial retention of form and meaning.

---

### 4. Semantic Similarity Layer
The pipeline computes sentence-level semantic similarity using **SBERT**.

- **SBERT similarity** captures broad semantic closeness between target and response
- this acts as a fast, robust semantic baseline

---

### 5. Deep Meaning Verification via NLI
The final semantic refinement stage uses a multilingual **Natural Language Inference (NLI)** model.

Instead of treating the task like generic similarity, NLI directly tests whether the learner response **preserves the meaning** of the target sentence.

This is especially well aligned with EIT scoring, where the central question is not just lexical overlap, but whether the response still entails the original message.

---

### 6. Interpretable Scoring Engine
For ambiguous responses, extracted features are combined using a weighted mathematical scoring function:

```text
Raw Score = w1 × NLI_margin + w2 × SBERT_similarity + w3 × Lemma_recall
```

### 7. QWK-Based Threshold Optimization
Rather than manually guessing score boundaries, the system tunes the threshold cutoffs using **Powell optimization** to maximize **Quadratic Weighted Kappa (QWK)** on the training split.

This produces empirically grounded boundaries between rubric levels:
- 0
- 1
- 2
- 3
- 4

---

### 8. A/B Scientific Design
To make the project more rigorous, AutoEIT includes two scoring approaches:

#### Approach A — Heuristic Scorer
A fully interpretable weighted feature scorer with optimized thresholds.

#### Approach B — Ordinal ML Model
A lightweight ordinal classifier trained on the extracted features as an experimental challenger.

This allows the system to compare:
- interpretability vs learned mapping
- deterministic thresholding vs data-driven ordinal classification

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Siddhazntx/AutoEIT-II
cd AutoEit_2
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv autoeit311
.\autoeit311\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv autoeit311
source autoeit311/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Spanish Model

```bash
python -m spacy download es_core_news_lg
```

---

## ▶️ Running the Pipeline

The project includes a command-line entry script for reproducible execution.

### Standard Run
```bash
python run_pipeline.py
```

### Debug Mode
```bash
python run_pipeline.py --debug
```

---

## 🧪 Testing and Sanity Checks

To run a quick sanity check on the pipeline components with sample data:

```bash
python test_run.py
```

This will load a few sample data points, show preprocessing steps, and verify that the feature extractors are working correctly.

---

## 📤 Exporting Results

To export the final predictions into a Excel output:

```bash
python export_results.py
```

---

## 📊 Notebooks

The project includes Jupyter notebooks for exploratory data analysis and evaluation:

- **`notebooks/data_exploration.ipynb`**: Exploratory Data Analysis (EDA) for understanding dataset distribution, class imbalance, and justifying architectural decisions.
- **`notebooks/final_evaluation.ipynb`**: Detailed evaluation of the winning model with statistical analysis, confusion matrices, and error analysis.

To run the notebooks, ensure you have Jupyter installed and the virtual environment activated:

```bash
jupyter notebook notebooks/
```

---

## 📁 Repository Structure

```
AutoEit_2/
├── configs/
│   └── config.yaml              # Centralized configuration file
├── data/
│   └── raw/                     # Input data files (Excel/CSV)
├── notebooks/
│   ├── data_exploration.ipynb   # EDA and dataset profiling
│   └── final_evaluation.ipynb   # Model evaluation and analysis
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Master orchestration pipeline
│   ├── data/
│   │   └── data_loader.py       # Data ingestion layer
│   ├── preprocessing/
│   │   └── preprocessor.py      # Cleaning + early quality gate
│   ├── features/                # Linguistic, SBERT, NLI extractors
│   │   ├── __init__.py
│   │   ├── cross_encoder.py
│   │   ├── feature_extractor.py
│   │   ├── linguistic.py
│   │   ├── nli_scorer.py
│   │   └── sbert.py
│   └── scoring/                 # Heuristic scorer, threshold optimizer, ordinal model
│       ├── heuristic_scorer.py
│       ├── ordinal_model.py
│       └── thresholding.py
├── export_results.py            # Utility for exporting final reports
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main CLI entry point
└── test_run.py                  # Sanity check script
```

**Note:** Generated outputs (in `data/processed/` and `data/cache/`), virtual environments, and temporary files are not included in the repository and should be added to `.gitignore`.

---

## 🔧 Configuration

The pipeline is configured via `configs/config.yaml`. This file contains parameters for:

- Data paths and preprocessing settings
- Feature extraction hyperparameters
- Scoring weights and thresholds
- Model training parameters

Modify this file to customize the pipeline behavior.

---


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
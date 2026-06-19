# 🌾 High-Precision Crop Recommendation — Explainable Fuzzy Ensemble Genetic Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](notebooks/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bhavesh2k5/High-Precision-Crop-Recommendation-ML-model/blob/main/notebooks/02_model_pipeline.ipynb)

> A multi-phase machine learning framework that combines **Genetic Algorithm optimisation**, **Fuzzy Logic reasoning**, and **Explainable AI (SHAP + LIME)** to recommend the optimal crop for given soil and climate conditions with high accuracy and full interpretability.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Crop Classes](#-crop-classes)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Traditional crop recommendation systems rely on simple rule-based logic or single classifiers that lack explainability. This project proposes a novel hybrid framework that:

1. **Optimises a weighted ensemble** of Decision Tree, Random Forest, and XGBoost using a **Genetic Algorithm (DEAP)** — finding the ideal model blend automatically.
2. **Layered with Fuzzy Logic** (scikit-fuzzy) for interpretable linguistic reasoning over soil/climate inputs.
3. **Explained with SHAP & LIME** to show *why* a specific crop was recommended.

---

## 🏗️ Pipeline Architecture

```
Raw Data (Train Dataset.csv)
        │
        ▼
┌─────────────────────────────────┐
│  Phase 1 · Data Cleaning        │  Deduplication, outlier removal,
│  01_data_cleaning.ipynb         │  column standardisation
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Feature Engineering            │  n_p_ratio, n_k_ratio, temp_rainfall
│  + SMOTE Oversampling           │  → balanced combined_dataset.csv
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 2 · Baseline Models      │  Decision Tree │ Random Forest │ XGBoost
│  02_model_pipeline.ipynb        │  (establishes "before" performance)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 3 · Fuzzy Framework      │  6 antecedents (N, P, K, pH,
│                                 │  rainfall, temp) with membership
│                                 │  functions (Low / Medium / High)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 4 · Genetic Algorithm    │  DEAP evolves optimal weights
│           Optimisation          │  [dt_depth, rf_est, rf_depth,
│                                 │   xgb_est, xgb_lr, w1, w2, w3]
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 5 · Genetic-Fusion       │  GA-optimised VotingClassifier
│           Ensemble              │  (champion_model)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 6 · Explainable AI       │  SHAP global feature importance
│           (SHAP + LIME)         │  LIME per-prediction explanation
└────────────────┬────────────────┘
                 │
                 ▼
        Crop Recommendation
          + Fuzzy Advice
```

---

## 📊 Dataset

Two datasets are included in the `data/` directory:

| File | Rows | Columns | Description |
|---|---|---|---|
| `Train Dataset.csv` | 18,079 | 8 | Raw training data — N, P, K, pH, rainfall, temperature, Crop |
| `SMOTE Test Dataset.csv` | 21,360 | 10 | Cleaned + SMOTE-balanced test set with engineered features |
| `combined_dataset.csv` | 21,360 | 10 | Output of `01_data_cleaning.ipynb` (same as SMOTE Test Dataset) |

### Feature Descriptions

| Feature | Unit | Description |
|---|---|---|
| `N` / `n` | kg/ha | Nitrogen content of soil |
| `P` / `p` | kg/ha | Phosphorus content of soil |
| `K` / `k` | kg/ha | Potassium content of soil |
| `pH` / `ph` | — | Soil acidity/alkalinity (0–14) |
| `rainfall` | mm | Average annual rainfall |
| `temperature` | °C | Average annual temperature |
| `n_p_ratio` | — | Engineered: N / (P + 1) |
| `n_k_ratio` | — | Engineered: N / (K + 1) |
| `temp_rainfall` | — | Engineered: temperature × rainfall |

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Baseline Decision Tree | ~85% |
| Baseline Random Forest | ~91% |
| Baseline XGBoost | ~93% |
| **GA-Optimised Genetic-Fusion Ensemble** | **~99%** |

> The Genetic Algorithm discovers the optimal hyperparameters *and* voting weights for the three constituent models, achieving a significant accuracy lift over any single baseline.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Bhavesh2k5/High-Precision-Crop-Recommendation-ML-model.git
cd High-Precision-Crop-Recommendation-ML-model
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebooks in order

**Step 1 — Data Cleaning:**
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
```
Run all cells. This generates `data/combined_dataset.csv`.

**Step 2 — Model Pipeline:**
```bash
jupyter notebook notebooks/02_model_pipeline.ipynb
```
Run all cells to train the full Genetic-Fuzzy Ensemble and view SHAP/LIME explanations.

---

## 📁 Project Structure

```
├── data/
│   ├── Train Dataset.csv          # Raw training dataset
│   ├── SMOTE Test Dataset.csv     # Cleaned & balanced test dataset
│   └── combined_dataset.csv       # Output of 01_data_cleaning.ipynb
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb     # Data cleaning, feature engineering, SMOTE
│   └── 02_model_pipeline.ipynb    # Full model training pipeline (Phases 1-7)
│
├── src/
│   └── predict.py                 # Standalone inference script / importable API
│
├── results/                       # Output figures from model notebooks
│
├── requirements.txt               # Pinned Python dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## 💻 Usage

### Run inference from the command line
```bash
python src/predict.py --N 90 --P 42 --K 43 --pH 6.5 --rainfall 202 --temperature 24
```

### Import as a module
```python
from src.predict import predict_crop, engineer_features

# After loading your trained model and label encoder from the notebook:
result = predict_crop(
    N=90, P=42, K=43,
    pH=6.5, rainfall=202, temperature=24,
    model=champion_model,
    label_encoder=le
)

print(f"Recommended Crop : {result['crop']}")
print(f"Confidence       : {result['confidence']}%")
print(f"Top 3 Crops      : {result['top3']}")
print(f"Fuzzy Advice     :")
for tip in result['fuzzy_advice']:
    print(f"  • {tip}")
```

**Sample output:**
```
Recommended Crop : rice
Confidence       : 98.7%
Top 3 Crops      : [('rice', 98.7), ('jute', 0.8), ('maize', 0.5)]
Fuzzy Advice     :
  • Nitrogen is HIGH — reduce nitrogenous fertiliser application.
  • Rainfall is HIGH — ensure good drainage to prevent waterlogging.
```

---

## 🔬 How It Works

### Genetic Algorithm (Phase 4)
The DEAP library evolves a population of 12 candidate "fusion models". Each individual's **DNA** encodes:
- `dt_depth` — Decision Tree max depth
- `rf_estimators`, `rf_depth` — Random Forest parameters
- `xgb_estimators`, `xgb_depth`, `xgb_lr` — XGBoost parameters
- `w1, w2, w3` — voting weights for DT, RF, XGBoost respectively

Fitness = validation accuracy of the assembled `VotingClassifier`. The GA runs for 3 generations with crossover probability 0.7 and mutation probability 0.3.

### Fuzzy Logic Layer (Phase 3)
Each of the 6 input features is mapped to linguistic categories (Low / Medium / High) using trapezoidal membership functions. The fuzzy inference system computes a crop suitability score alongside the ML prediction, enabling human-understandable reasoning.

### SHAP + LIME (Phase 6)
- **SHAP** provides global feature importance showing *which features drive predictions overall*.
- **LIME** provides local explanations showing *why this specific sample was predicted as this crop*.

---

## 🌱 Crop Classes

The model recommends one of **22 crops**:

| | | | |
|---|---|---|---|
| apple | banana | blackgram | chickpea |
| coconut | coffee | cotton | grapes |
| jute | kidneybeans | lentil | maize |
| mango | mothbeans | mungbean | muskmelon |
| orange | papaya | pigeonpeas | pomegranate |
| rice | watermelon | | |

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{bhavesh2026cropfuzzyga,
  author    = {Bhavesh},
  title     = {An Explainable Fuzzy Ensemble Genetic Framework for High Precision Crop Recommendation},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Bhavesh2k5/High-Precision-Crop-Recommendation-ML-model}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

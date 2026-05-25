# Logistic Regression Classification — Project README

> Binary classification using Logistic Regression with hyperparameter tuning and cross-validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
  - [Baseline Model](#baseline-model)
  - [Tuned Model](#tuned-model)
  - [Comparison Summary](#comparison-summary)
- [Results & Conclusions](#results--conclusions)
- [What to Do Next](#what-to-do-next)
- [How to Run](#how-to-run)

---

## Project Overview

This project implements a **binary classification pipeline** using Logistic Regression to predict one of two target classes (Class 0 / Class 1). The pipeline covers:

1. Data preprocessing and splitting
2. Baseline Logistic Regression model training and evaluation
3. Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
4. Cross-validation to prevent overfitting and ensure generalization
5. Comparative performance analysis between baseline and tuned models

---

## Dataset

| Property | Value |
|---|---|
| Total samples (test set) | 300 |
| Class 0 samples | 135 (45%) |
| Class 1 samples | 165 (55%) |
| Class imbalance | Mild (55/45 split) |
| Task type | Binary classification |

> **Note:** The dataset has a mild class imbalance. Class 1 is slightly over-represented (55%), which is generally manageable for Logistic Regression without special resampling techniques, though it is worth monitoring.

---

## Project Structure

```
├── logistic.ipynb          # Main Jupyter Notebook with full pipeline
├── README.md               # Project documentation (this file)
└── data/                   # Dataset directory (add your data here)
```

---

## Dependencies

Install all required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Model building, tuning, and evaluation |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Confusion matrix heatmap and statistical plots |
| `jupyter` | Interactive notebook environment |

---

## Methodology

### 1. Data Preprocessing
- Handled missing values (if any)
- Feature scaling using `StandardScaler` (important for Logistic Regression as it is sensitive to feature magnitude)
- Train/test split (test set: 300 samples)

### 2. Baseline Logistic Regression
Trained a default `LogisticRegression()` from scikit-learn with no tuning. Evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Overall Accuracy

### 3. Hyperparameter Tuning
Applied grid search or randomized search over key Logistic Regression hyperparameters:

| Hyperparameter | Description | Common values explored |
|---|---|---|
| `C` | Inverse of regularization strength | `[0.001, 0.01, 0.1, 1, 10, 100]` |
| `solver` | Optimization algorithm | `['lbfgs', 'liblinear', 'saga']` |
| `penalty` | Regularization type | `['l1', 'l2', 'elasticnet']` |
| `max_iter` | Maximum iterations for convergence | `[100, 200, 500]` |

### 4. Cross-Validation
Used **k-fold cross-validation** (typically k=5 or k=10) to:
- Estimate model generalization performance reliably
- Prevent overfitting to any single train/test split
- Select the best hyperparameter combination via `GridSearchCV`

---

## Model Performance

### Baseline Model

```
Confusion Matrix:
[[118  17]
 [ 29 136]]
```

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.80 | 0.87 | 0.84 | 135 |
| 1 | 0.89 | 0.82 | 0.86 | 165 |
| **Accuracy** | | | **0.847** | 300 |
| Macro avg | 0.85 | 0.85 | 0.85 | 300 |
| Weighted avg | 0.85 | 0.85 | 0.85 | 300 |

**Breakdown:**
- True Negatives (TN): 118 — correctly identified Class 0
- False Positives (FP): 17 — Class 0 wrongly predicted as Class 1
- False Negatives (FN): 29 — Class 1 wrongly predicted as Class 0
- True Positives (TP): 136 — correctly identified Class 1

---

### Tuned Model

> After applying hyperparameter tuning and cross-validation.

```
Confusion Matrix:
[[124  11]
 [ 33 132]]
```

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.79 | 0.92 | 0.85 | 135 |
| 1 | 0.92 | 0.80 | 0.86 | 165 |
| **Accuracy** | | | **0.853** | 300 |
| Macro avg | 0.86 | 0.86 | 0.85 | 300 |
| Weighted avg | 0.86 | 0.85 | 0.85 | 300 |

**Breakdown:**
- True Negatives (TN): 124 (+6 vs baseline)
- False Positives (FP): 11 (−6 vs baseline, **35% reduction**)
- False Negatives (FN): 33 (+4 vs baseline)
- True Positives (TP): 132 (−4 vs baseline)

---

### Comparison Summary

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Accuracy | 84.67% | 85.33% | ▲ +0.67% |
| Class 0 Precision | 0.80 | 0.79 | ▼ −0.01 |
| Class 0 Recall | 0.87 | 0.92 | ▲ +0.05 |
| Class 1 Precision | 0.89 | 0.92 | ▲ +0.03 |
| Class 1 Recall | 0.82 | 0.80 | ▼ −0.02 |
| Macro F1 | 0.85 | 0.85 | — no change |
| False Positives | 17 | 11 | ▲ −35% |
| False Negatives | 29 | 33 | ▼ +14% |

---

## Results & Conclusions

### 1. Overall Performance
Both models achieve ~85% accuracy on the 300-sample test set, which is a solid result for Logistic Regression as a linear classifier. The tuned model shows a marginal but consistent improvement.

### 2. Precision–Recall Trade-off (Key Finding)
Hyperparameter tuning introduced a **precision-recall trade-off**. The optimizer found a configuration (likely a higher regularization value for `C`) that raised the decision threshold for predicting Class 1:

- **Benefit:** False positives dropped sharply (17 → 11, a 35% reduction). The model is now much more confident when it predicts Class 1.
- **Cost:** False negatives increased slightly (29 → 33). Four additional Class 1 cases are now missed.

This is expected behaviour — to be more precise, the model requires stronger evidence to commit to a positive prediction, which causes some true positives to fall below the threshold.

### 3. Class 0 vs Class 1 Behaviour
- The tuned model became notably better at correctly identifying Class 0 cases (recall 0.87 → 0.92).
- Class 1 recall dipped slightly (0.82 → 0.80), but Class 1 precision improved (0.89 → 0.92).

### 4. Performance Ceiling
The near-identical macro F1 score (0.85 in both models) and small accuracy delta (+0.67%) suggest the model is **approaching its performance ceiling** for a linear classifier on this dataset. Logistic Regression may not be able to capture non-linear decision boundaries that could improve results further.

### 5. Domain Consideration
Whether the tuned model is better depends on the business/domain context:

| Scenario | Preferred model | Reason |
|---|---|---|
| False positives are costly (e.g. spam filters, fraud alerts) | **Tuned** | Fewer false alarms |
| False negatives are costly (e.g. disease detection, fault detection) | **Baseline** | Higher Class 1 recall |

---

## What to Do Next

### Immediate improvements (no retraining needed)
- **Threshold tuning:** Plot the ROC curve and Precision-Recall curve. Move the classification threshold away from 0.5 to optimise for your specific FP/FN cost. This is free performance.
- **`class_weight='balanced'`:** Try setting this in `LogisticRegression()` to handle the mild class imbalance automatically.

### Model improvements
- **Try non-linear models:** Logistic Regression is limited to linear decision boundaries. Try:
  - `RandomForestClassifier`
  - `GradientBoostingClassifier` / `XGBClassifier`
  - `SVC(kernel='rbf')`
- **Feature engineering:** Create interaction terms or polynomial features to give the linear model more signal without switching algorithms.

### Evaluation improvements
- **Examine CV fold variance:** If fold scores vary widely (e.g. 0.78 to 0.91), the model is unstable — more data or stronger regularization is needed.
- **Error analysis:** Manually inspect the 33 false negatives and 11 false positives. Identifying a common pattern in misclassified samples often reveals the fastest path to improvement.
- **Learning curve:** Plot training vs validation accuracy as a function of training set size to diagnose underfitting vs overfitting.

---

## How to Run

1. **Clone or download the repository**
   ```bash
   git clone <your-repo-url>
   cd <project-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook logistic.ipynb
   ```

4. **Run all cells** in order:
   - Data loading and preprocessing
   - Baseline model training and evaluation
   - Hyperparameter tuning with cross-validation
   - Performance comparison and visualizations

---

## Author

> Add your name, contact, and project date here.

---

## License

> Specify your license here (e.g. MIT, Apache 2.0, or proprietary).

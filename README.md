# Applied Machine Learning & Deep Learning for Chronic Disease Stratification
### Benchmarking Traditional Machine Learning vs. Deep Learning on the CDC BRFSS / UCI Diabetes Health Indicators Dataset

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview
This repository contains the end-to-end machine learning and deep learning experimentation pipeline for the **Machine Learning Summative Final Project**. The investigation addresses the clinical screening challenge of Type 2 Diabetes risk classification utilizing the **CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) / UCI Machine Learning Repository (Dataset ID: 891)**.

The investigation rigorously compares **Traditional Machine Learning (Scikit-Learn)** against **Deep Learning (TensorFlow / Keras Functional API with `tf.data` input pipelines)** across 8 systematically varied experiments. We place particular emphasis on **class imbalance mitigation**, **cost-sensitive optimization**, **multi-mechanism regularization**, and **clinical decision threshold calibration** to maximize patient recall while managing false positive rates.

---

## Submission Artifact Links
- **GitHub Repository**: `[Insert Your GitHub Repository URL Here]`
- **Google Drive Submission Assets**: `[Insert Your Public Google Drive Folder Link Here]`
  - Contains: `raw_dataset.zip` (full 253,680 rows raw survey data), `best_classical_ml_model.joblib`, `best_deep_learning_model.keras`, `best_deep_learning_model.h5`, and `scaler.joblib`.
- **Recorded Presentation Video**: `[Insert Your 5-10 Minute Presentation Link Here]`

---

## Dataset Description & Compliance
- **Origin**: Sourced directly from the official [UCI Machine Learning Repository (ID: 891)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) and the US Centers for Disease Control and Prevention (CDC).
- **Scale**: **253,680 raw observations** (well exceeding the project threshold of 20,000 samples).
- **Features**: 21 original demographic, clinical, and behavioral attributes + 5 domain-engineered physiological indices.
- **Target Variable**: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes / Diabetes).
- **Imbalance**: 86.07% Negative (218,334), 13.93% Positive (35,346).
- **Data Policy**: Raw dataset files are strictly excluded from git tracking (`.gitignore`) and archived inside Google Drive as requested.

---

## Repository Structure
```text
diabetes_prediction_project/
│
├── data/
│   ├── raw/                  # Downloaded raw dataset & zip archive (excluded from git)
│   └── processed/            # Stratified train/val/test partitions & fitted scaler
│
├── models/                   # Serialized best model weights
│   ├── best_classical_ml_model.joblib
│   ├── best_deep_learning_model.keras
│   └── best_deep_learning_model.h5
│
├── figures/                  # Publication-quality evaluation visualizations
│   ├── learning_curves.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── threshold_optimization.png
│
├── logs/                     # Experimentation tracking & TensorBoard event logs
│   ├── fit/                  # TensorBoard execution runs
│   ├── experiment_results.csv
│   └── experiment_results.json
│
├── notebooks/                # Modular, self-contained demonstration notebook
│   └── diabetes_classification_experiments.ipynb
│
├── src/                      # Modular Python source modules
│   ├── data_loader.py        # Dataset fetching and validation
│   ├── preprocessing.py      # Feature engineering and stratified scaling
│   ├── models.py             # Scikit-Learn factories & tf.data Keras architectures
│   └── train_and_evaluate.py # Master 8-experiment execution script
│
├── final_academic_report.md  # Complete 4,500+ word IEEE formatted research paper
├── presentation_script.md    # 5-10 minute presentation slide deck & defense transcript
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project documentation
```

---

## Summary of 8 Systematic Experiments

All experiments were executed on an identical stratified split (177,576 training, 38,052 validation, 38,052 test) with fixed random seeds (`SEED=42`).

| Exp ID | Paradigm | Model Architecture | Key Hyperparameters / Settings | Optimization & Strategy | Test Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | F1-Score | F2-Score (Clinical) | ROC-AUC | PR-AUC |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | Traditional ML | Logistic Regression | $C=1.0$, L2 penalty, solver=`lbfgs`, $t=0.5$ | Convex optimization minimizing regularized log-loss | 0.8639 | 0.5695 | 0.5391 | 0.1613 | 0.2483 | 0.1875 | 0.8219 | 0.4040 |
| **Exp 2** | Traditional ML | Random Forest | $N=100$, `max_depth`=12, `min_samples_split`=20 | Bagging ensemble capturing non-linear interactions | 0.8664 | 0.5558 | 0.5977 | 0.1252 | 0.2071 | 0.1488 | 0.8252 | 0.4223 |
| **Exp 3** | Traditional ML | HistGradientBoosting | `lr`=0.08, `max_iter`=150, `max_leaf_nodes`=31 | Histogram-binned gradient boosted decision trees | 0.8662 | 0.5672 | 0.5749 | 0.1528 | 0.2414 | 0.1791 | 0.8293 | 0.4279 |
| **Exp 4** | Traditional ML | Cost-Sensitive Boosting | `class_weight`='balanced', `lr`=0.08, `max_iter`=150 | Cost-sensitive loss weighting inverse to class prevalence | 0.7215 | 0.7535 | 0.3075 | **0.7978** | 0.4439 | **0.6049** | **0.8296** | **0.4284** |
| **Exp 5** | Deep Learning | Baseline MLP (`tf.data`) | Dense(64) $\to$ Dense(32) $\to$ Dense(1), Adam `lr`=1e-3 | Standard feedforward network trained via `tf.data` | 0.8663 | 0.5526 | 0.6047 | 0.1177 | 0.1970 | 0.1403 | 0.8286 | 0.4240 |
| **Exp 6** | Deep Learning | Regularized Deep MLP | 128 $\to$ 64 $\to$ 32, BatchNorm, Dropout(0.3, 0.2), L2($10^{-4}$) | Multi-tiered regularization eliminating overfitting | 0.8661 | 0.5706 | 0.5687 | 0.1609 | 0.2508 | 0.1878 | 0.8287 | 0.4241 |
| **Exp 7** | Deep Learning | Cost-Weighted Deep MLP | `ExponentialDecay` LR, `class_weight`={0:1, 1:6.18}, EarlyStopping | Dynamic learning rate decay with cost-weighted loss | 0.7089 | **0.7546** | 0.3001 | **0.8180** | 0.4391 | **0.6081** | 0.8292 | 0.4237 |
| **Exp 8** | Deep Learning | Threshold-Optimized MLP | Architecture of Exp 6, Tuned Threshold $t=0.13$ | Post-hoc decision boundary calibration maximizing $F_2$ | 0.6919 | 0.7530 | 0.2902 | **0.8378** | 0.4311 | **0.6083** | 0.8287 | 0.4241 |

---

## Key Experimental Takeaways

1. **The Fallacy of Raw Classification Accuracy**: In an imbalanced epidemiology setting (13.9% positive class), unweighted models achieve ~86.6% accuracy by predicting the majority class, yet fail to detect over 83% of diabetic individuals (Recall: 11.7% - 16.1%).
2. **The Power of Loss Reweighting**: Applying cost-sensitive loss weighting in **Exp 4 (Gradient Boosting)** and **Exp 7 (Deep Learning)** dramatically elevates Recall to **79.8%** and **81.8%** respectively, improving the clinical screening metric ($F_2$-Score) from 0.18 to >0.60.
3. **Clinical Decision Threshold Calibration (Exp 8)**: Shifting the classification threshold to $t=0.13$ based on validation set optimization elevates Test Recall to **83.78%**, catching more than 4 out of 5 patients at risk without requiring model retraining.
4. **Computational Efficiency vs. Neural Modeling**: While Deep Learning demonstrated superior recall when paired with threshold calibration, Scikit-Learn's HistGradientBoosting proved significantly faster to train while matching neural network discriminative capacity (ROC-AUC ~0.83).

---

## Installation & Reproducibility Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/diabetes_prediction_project.git
cd diabetes_prediction_project
```

### 2. Create Virtual Environment
```bash
# Using Python venv
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Complete Pipeline
To reproduce all 8 experiments, train all models, log to TensorBoard, generate publication figures, and bundle Google Drive assets:
```bash
# Ingest raw dataset from UCI
python src/data_loader.py

# Preprocess and partition data
python src/preprocessing.py

# Execute 8 experiments and generate evaluation suite
python src/train_and_evaluate.py
```

### 5. Launch TensorBoard
To inspect real-time training dynamics, loss divergence, and ROC-AUC convergence across epochs:
```bash
tensorboard --logdir logs/fit
```

### 6. Run the Interactive Notebook
```bash
jupyter notebook notebooks/diabetes_classification_experiments.ipynb
```

---

## Citation & Academic References
Please refer to `final_academic_report.md` for the complete 4,500+ word academic research paper formatted in IEEE style, including detailed theoretical grounding, bias-variance analysis, and 12 peer-reviewed scholarly citations.

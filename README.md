# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F1_0.88-F7931E.svg?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_TPE-2962FF.svg?style=flat)](https://optuna.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)

**PES University | Department of Computer Science & Engineering (RR Campus)**  
**Course:** UE24CS352A Machine Learning  
**Project:** PathSnap Distributed Map-Matching & Route Reconstruction

---

## 📌 Executive Summary

GPS tracking systems often suffer from significant noise, urban canyon multipath interference, satellite dropouts, and low sampling frequencies. **PathSnap** is an end-to-end distributed machine learning system that accurately reconstructs vehicle trajectories by matching noisy GPS observations to a graph-based road network.

PathSnap implements:
1. **Candidate Generation & Feature Engineering:** KD-Tree spatial indexing with 8 geometric & kinematic features.
2. **Multi-Model Benchmarking:** 5 model families evaluated with 3-Fold GroupKFold cross-validation across 80 training trips and 10 student-testing trips.
3. **Bayesian Hyperparameter Optimization:** Optuna Tree-structured Parzen Estimator (TPE) tuning the selected `HistGradientBoosting` architecture.
4. **HMM & Viterbi Decoding:** Sequence-level decoding enforcing topological head-to-tail road connectivity, improving route continuity from **86.57% to 98.26%**.
5. **Two-Laptop Distributed Deployment:** FastAPI server (Laptop 1) housing the models and protected ground-truth, communicating over LAN/WiFi with a modern evaluator interface (Laptop 2).

---

## 📁 Repository Structure

```
PathSnap/
├── api/
│   └── main.py                     # Laptop 1: FastAPI server with inference & evaluation endpoints
├── ui/
│   └── index.html                  # Laptop 2: Evaluator interface with live map rendering
├── src/
│   ├── core_utils.py               # Data loaders, metric projections, spatial KD-Tree candidate extraction
│   ├── data_processing.py          # Unified preprocessing for 80 train + 10 test trips
│   ├── hmm_viterbi.py              # HMM emission & transition modeling + Viterbi recurrence
│   ├── train_models.py             # Trains & benchmarks 5 algorithm families
│   ├── bayesian_opt.py             # Optuna Bayesian hyperparameter search
│   ├── evaluate_route.py           # Classifier vs HMM route reconstruction & map generator
│   └── generate_report.py          # ReportLab technical report compiler
├── models/
│   ├── final_tuned_model.pkl       # Optuna-tuned HistGradientBoosting model artifact
│   ├── DecisionTree.pkl
│   ├── GaussianNB.pkl
│   ├── HistGradientBoosting.pkl
│   ├── LogisticRegression.pkl
│   └── RandomForest.pkl
├── visualizations/
│   ├── trip_00000000_map.html      # Interactive Folium map for test trip 00000000
│   ├── trip_00000007_map.html      # Interactive Folium map for test trip 00000007
│   └── trip_00000040_map.html      # Interactive Folium map for test trip 00000040
├── results/
│   ├── model_comparison.json       # Benchmark metrics across all 5 models
│   ├── model_comparison.png        # Comparative performance plots
│   ├── bayesian_optimization.json  # Search configurations and trial records
│   ├── optuna_optimization_history.png
│   └── route_reconstruction_results.json
├── report/
│   └── PathSnap_Technical_Report.pdf # Final publication-quality consolidated PDF report
├── requirements.txt
└── README.md
```

---

## 📊 Experimental Results

### 1. Overall Model Comparison (Across Algorithm Families)

| Model Name | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Infer Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | C=1.0, StandardScaler, max_iter=300 | 0.8107 | 72.69% | 0.8388 | 0.1538 | 29.6 ms |
| **Gaussian Naive Bayes** | Probabilistic | var_smoothing=1e-8, StandardScaler | 0.6523 | 70.65% | 0.5889 | 1.2490 | 82.7 ms |
| **Decision Tree** | Tree-based | max_depth=12, min_samples_leaf=20 | 0.8499 | 73.45% | 0.8762 | 0.1295 | 31.2 ms |
| **Random Forest** | Ensemble (Bagging) | n_estimators=60, max_depth=14 | 0.8555 | 73.90% | 0.8823 | 0.1102 | 254.7 ms |
| **HistGradientBoosting** | Gradient Boosting | max_iter=100, max_depth=8, lr=0.1 | **0.8553** | **73.80%** | **0.8831** | **0.1096** | **362.5 ms** |

*Note: In accordance with hackathon requirements, `HistGradientBoosting` was selected for Bayesian hyperparameter optimization due to lowest log-loss and superior probability calibration.*

### 2. Bayesian Hyperparameter Optimization (Optuna TPE)

- **Search Space:** `max_iter` [60, 160], `max_depth` [5, 12], `learning_rate` [0.03, 0.25], `min_samples_leaf` [10, 60], `l2_regularization` [1e-4, 10.0].
- **Objective:** 3-Fold GroupKFold Cross-Validation Macro F1 on 80 training trips.
- **Optimal Hyperparameters:** `max_iter=100`, `max_depth=9`, `learning_rate=0.0739`, `min_samples_leaf=50`, `l2_regularization=0.000113`.
- **Tuned Model Performance:** 
  - Cross-Validation Macro F1: **0.8594**
  - Student-Test Top-1 Observation Accuracy: **73.89%**
  - Student-Test Macro F1: **0.8829**
  - Log Loss: **0.1093**

### 3. Route Reconstruction: Classifier-Only vs Classifier + HMM Viterbi

| Metric | Classifier-Only Baseline | Classifier + HMM / Viterbi | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Mean Pointwise Accuracy** | 75.89% | 66.69% | -9.20% |
| **Mean Route Continuity** | 86.57% | **98.26%** | **+11.69%** |
| **Average Latency per Trip** | 1.1 ms | 185.0 ms | Real-time (<0.2s) |

*Sequential Viterbi decoding strictly penalizes disconnected edge transitions, raising topological route continuity to **98.26%** across all student-testing trajectories.*

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Reproduce Data Preprocessing, Training & Optimization
```bash
# 1. Extract candidates & features
python src/data_processing.py

# 2. Train and benchmark the 5 models
python src/train_models.py

# 3. Run Bayesian hyperparameter optimization
python src/bayesian_opt.py

# 4. Evaluate route reconstruction & generate Folium maps
python src/evaluate_route.py

# 5. Compile the comprehensive PDF Technical Report
python src/generate_report.py
```

---

## 💻 Two-Laptop Evaluator Deployment

### Laptop 1: Model & API Service Host
1. Start the FastAPI server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
2. Note your local IP (e.g., `http://192.168.1.100:8000` or `http://localhost:8000`).
3. View Swagger API Docs: `http://localhost:8000/docs`

### Laptop 2: Evaluator User Interface
1. Open `ui/index.html` in any web browser (Chrome, Edge, Firefox).
2. Enter the Laptop 1 IP/URL in the connection pill and click **Connect**.
3. Select any of the 10 evaluation trips from the left sidebar.
4. Click **Run Reconstruction Pipeline**.
5. Inspect the real-time latency, route continuity, and interactive Folium map visualization.

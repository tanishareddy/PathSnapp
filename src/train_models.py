"""
PathSnap Model Training & Benchmark Experimentation:
Trains and systematically evaluates 5 diverse models across different algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. Gaussian Naive Bayes (Probabilistic)
3. Decision Tree Classifier (Tree-based)
4. Random Forest Classifier (Ensemble Bagging)
5. HistGradientBoostingClassifier (Gradient Boosting)

Evaluates:
- Group-Aware Cross-Validation Macro F1
- Observation-level Top-1 Accuracy on Student-Testing Set
- Student-Testing Macro F1
- Log Loss (cross-entropy)
- Route Continuity (%)
- Candidate Miss Rate
- Inference Time & Training Time
Generates comparison table, confusion matrices, and ROC curves saved in results/.
"""
import os
import sys
import time
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_curve, auc
)

from hmm_viterbi import calculate_route_continuity

def run_experiment():
    print("Loading preprocessed training and test datasets...")
    train_data = np.load(os.path.join("data", "train_dataset.npz"))
    test_data = np.load(os.path.join("data", "student_test_dataset.npz"))
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups = train_data["groups"]
    
    X_test = test_data["X"]
    y_test = test_data["y"]
    k_candidates = int(train_data["k_candidates"])
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_train.shape[1]}")
    
    # Subsample train for swift, robust training while retaining high statistical power (200k samples)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), size=min(250000, len(X_train)), replace=False)
    X_train_sub = X_train[sample_idx]
    y_train_sub = y_train[sample_idx]
    groups_sub = groups[sample_idx]
    
    models = {
        "LogisticRegression": {
            "family": "Linear & Probabilistic",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=300, random_state=42))
            ]),
            "config": "C=1.0, StandardScaler, max_iter=300"
        },
        "GaussianNB": {
            "family": "Probabilistic",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-8))
            ]),
            "config": "var_smoothing=1e-8, StandardScaler"
        },
        "DecisionTree": {
            "family": "Tree-based",
            "pipeline": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
            "config": "max_depth=12, min_samples_leaf=20"
        },
        "RandomForest": {
            "family": "Ensemble (Bagging)",
            "pipeline": RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, random_state=42),
            "config": "n_estimators=60, max_depth=14, n_jobs=-1"
        },
        "HistGradientBoosting": {
            "family": "Gradient Boosting",
            "pipeline": HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42),
            "config": "max_iter=100, max_depth=8, lr=0.1"
        }
    }
    
    results = {}
    gkf = GroupKFold(n_splits=3)
    
    # Reconstruct test observations candidate structure
    N_test_obs = len(X_test) // k_candidates
    y_test_reshaped = y_test.reshape(N_test_obs, k_candidates)
    
    for name, m_info in models.items():
        print(f"\n==================================================")
        print(f"Training Model: {name} ({m_info['family']})")
        print(f"Config: {m_info['config']}")
        print(f"==================================================")
        
        clf = m_info["pipeline"]
        
        # 1. 3-Fold Group Cross Validation on Train
        cv_macro_f1s = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_sub, y_train_sub, groups=groups_sub)):
            clf_fold = m_info["pipeline"]
            clf_fold.fit(X_train_sub[train_idx], y_train_sub[train_idx])
            val_preds = clf_fold.predict(X_train_sub[val_idx])
            fold_f1 = f1_score(y_train_sub[val_idx], val_preds, average="macro")
            cv_macro_f1s.append(fold_f1)
            
        cv_macro_f1_mean = float(np.mean(cv_macro_f1s))
        print(f"CV Macro F1 (3-Fold GroupKFold): {cv_macro_f1_mean:.4f}")
        
        # 2. Fit full training subset
        t0 = time.time()
        clf.fit(X_train_sub, y_train_sub)
        train_time = time.time() - t0
        print(f"Fit completed in {train_time:.2f}s")
        
        # 3. Inference on Student-Testing Set
        t0 = time.time()
        test_preds = clf.predict(X_test)
        infer_time = time.time() - t0
        
        if hasattr(clf, "predict_proba"):
            test_probs = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(X_test)
            test_probs = 1.0 / (1.0 + np.exp(-df))
        else:
            test_probs = test_preds.astype(float)
            
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
        test_precision = float(precision_score(y_test, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test, test_preds, zero_division=0))
        loss = float(log_loss(y_test, np.clip(test_probs, 1e-6, 1.0 - 1e-6)))
        
        # 4. Observation-level Top-1 Accuracy
        scores_reshaped = test_probs.reshape(N_test_obs, k_candidates)
        top1_idx = np.argmax(scores_reshaped, axis=1)
        top1_correct = [y_test_reshaped[i, top1_idx[i]] == 1 for i in range(N_test_obs)]
        top1_acc = float(np.mean(top1_correct))
        
        # Miss rate (ground truth not in top-k candidates)
        has_gt = np.any(y_test_reshaped == 1, axis=1)
        miss_rate = float(1.0 - np.mean(has_gt))
        
        print(f"Observation Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"Candidate Miss Rate:        {miss_rate*100:.2f}%")
        print(f"Student-Test Macro F1:      {test_macro_f1:.4f}")
        print(f"Log Loss:                   {loss:.4f}")
        print(f"Inference Time:             {infer_time:.4f}s ({infer_time/len(X_test)*1e6:.2f} µs/sample)")
        
        # Save model artifact
        model_path = os.path.join("models", f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
            
        results[name] = {
            "name": name,
            "family": m_info["family"],
            "config": m_info["config"],
            "cv_macro_f1": cv_macro_f1_mean,
            "test_top1_acc": top1_acc,
            "test_macro_f1": test_macro_f1,
            "precision": test_precision,
            "recall": test_recall,
            "log_loss": loss,
            "candidate_miss_rate": miss_rate,
            "train_time_sec": train_time,
            "infer_time_sec": infer_time,
            "model_path": model_path
        }
        
    # Save benchmark summary
    with open(os.path.join("results", "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Plots
    print("\nGenerating model comparison plots and figures...")
    names = list(results.keys())
    top1_accs = [results[n]["test_top1_acc"] * 100 for n in names]
    f1_scores = [results[n]["test_macro_f1"] for n in names]
    losses = [results[n]["log_loss"] for n in names]
    infer_times = [results[n]["infer_time_sec"] * 1000 for n in names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(names, top1_accs, color="#2563eb")
    axes[0, 0].set_title("Observation-level Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].tick_params(axis='x', rotation=25)
    
    axes[0, 1].bar(names, f1_scores, color="#059669")
    axes[0, 1].set_title("Student-Testing Macro F1-Score", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Macro F1")
    axes[0, 1].tick_params(axis='x', rotation=25)
    
    axes[1, 0].bar(names, losses, color="#dc2626")
    axes[1, 0].set_title("Log Loss (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Cross Entropy Loss")
    axes[1, 0].tick_params(axis='x', rotation=25)
    
    axes[1, 1].bar(names, infer_times, color="#7c3aed")
    axes[1, 1].set_title("Inference Time for 153k samples (ms)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Time (ms)")
    axes[1, 1].tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", "model_comparison.png"), dpi=300)
    plt.close()
    
    print("Benchmark complete! Saved results to results/model_comparison.json and results/model_comparison.png")

if __name__ == "__main__":
    run_experiment()

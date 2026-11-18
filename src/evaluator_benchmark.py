"""
Comprehensive Benchmarking of 5 Model Families on the Evaluator Test Dataset.
Evaluates:
- Binary Candidate Classification Accuracy (%)
- Observation-level Top-1 Accuracy (%)
- Observation-level Top-3 Accuracy (%)
- Macro F1-Score
- Candidate Miss Rate (%)
- Binary Cross Entropy Log Loss
- Inference Time
Generates comparison table and figures in results/.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

def run_evaluator_benchmark():
    print("Loading preprocessed datasets...")
    train_data = np.load(os.path.join("data", "train_dataset_v2.npz"))
    eval_data = np.load(os.path.join("data", "evaluator_test_dataset_v2.npz"))
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups = train_data["groups"]
    
    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    k_candidates = int(eval_data["k_candidates"])
    
    print(f"Train samples: {len(X_train)}, Evaluator samples: {len(X_eval)}, Features: {X_train.shape[1]}")
    
    # Subsample 250k for fast, balanced cross-validation
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
            "pipeline": HistGradientBoostingClassifier(max_iter=120, max_depth=9, learning_rate=0.08, min_samples_leaf=40, random_state=42),
            "config": "max_iter=120, max_depth=9, lr=0.08, min_leaf=40"
        }
    }
    
    results = {}
    gkf = GroupKFold(n_splits=3)
    
    N_eval_obs = len(X_eval) // k_candidates
    y_eval_reshaped = y_eval.reshape(N_eval_obs, k_candidates)
    
    for name, m_info in models.items():
        print(f"\n==================================================")
        print(f"Training Model: {name} ({m_info['family']})")
        print(f"Config: {m_info['config']}")
        print(f"==================================================")
        
        clf = m_info["pipeline"]
        
        # 1. 3-Fold Group CV on Train
        cv_macro_f1s = []
        for train_idx, val_idx in gkf.split(X_train_sub, y_train_sub, groups=groups_sub):
            clf.fit(X_train_sub[train_idx], y_train_sub[train_idx])
            val_preds = clf.predict(X_train_sub[val_idx])
            cv_macro_f1s.append(f1_score(y_train_sub[val_idx], val_preds, average="macro"))
            
        cv_f1 = float(np.mean(cv_macro_f1s))
        print(f"CV Macro F1 (3-Fold GroupKFold): {cv_f1:.4f}")
        
        # 2. Fit full training subset
        t0 = time.time()
        clf.fit(X_train_sub, y_train_sub)
        train_time = time.time() - t0
        
        # 3. Test on Evaluator Testing Trips
        t0 = time.time()
        eval_preds = clf.predict(X_eval)
        infer_time = time.time() - t0
        
        if hasattr(clf, "predict_proba"):
            eval_probs = clf.predict_proba(X_eval)[:, 1]
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(X_eval)
            eval_probs = 1.0 / (1.0 + np.exp(-df))
        else:
            eval_probs = eval_preds.astype(float)
            
        # Binary candidate classification accuracy
        candidate_binary_acc = float(accuracy_score(y_eval, eval_preds))
        macro_f1 = float(f1_score(y_eval, eval_preds, average="macro"))
        prec = float(precision_score(y_eval, eval_preds, zero_division=0))
        rec = float(recall_score(y_eval, eval_preds, zero_division=0))
        loss = float(log_loss(y_eval, np.clip(eval_probs, 1e-6, 1.0 - 1e-6)))
        
        # Observation-level Top-1 and Top-3 Accuracy
        scores_reshaped = eval_probs.reshape(N_eval_obs, k_candidates)
        top1_idx = np.argmax(scores_reshaped, axis=1)
        top1_correct = [y_eval_reshaped[i, top1_idx[i]] == 1 for i in range(N_eval_obs)]
        top1_acc = float(np.mean(top1_correct))
        
        # Top-3 Accuracy
        top3_idx = np.argsort(-scores_reshaped, axis=1)[:, :3]
        top3_correct = [np.any(y_eval_reshaped[i, top3_idx[i]] == 1) for i in range(N_eval_obs)]
        top3_acc = float(np.mean(top3_correct))
        
        miss_rate = float(1.0 - np.mean(np.any(y_eval_reshaped == 1, axis=1)))
        
        print(f"Candidate Binary Classification Accuracy: {candidate_binary_acc*100:.2f}%")
        print(f"Observation Top-1 Accuracy:                {top1_acc*100:.2f}%")
        print(f"Observation Top-3 Accuracy:                {top3_acc*100:.2f}%")
        print(f"Candidate Miss Rate:                       {miss_rate*100:.2f}%")
        print(f"Evaluator-Test Macro F1:                   {macro_f1:.4f}")
        print(f"Log Loss:                                  {loss:.4f}")
        print(f"Inference Time:                            {infer_time:.4f}s")
        
        # Save model
        with open(os.path.join("models", f"{name}_v2.pkl"), "wb") as f:
            pickle.dump(clf, f)
            
        results[name] = {
            "name": name,
            "family": m_info["family"],
            "config": m_info["config"],
            "cv_macro_f1": cv_f1,
            "candidate_binary_acc": candidate_binary_acc,
            "eval_top1_acc": top1_acc,
            "eval_top3_acc": top3_acc,
            "eval_macro_f1": macro_f1,
            "precision": prec,
            "recall": rec,
            "log_loss": loss,
            "candidate_miss_rate": miss_rate,
            "train_time_sec": train_time,
            "infer_time_sec": infer_time
        }
        
    with open(os.path.join("results", "evaluator_test_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nBenchmark on Evaluator Test complete! Saved to results/evaluator_test_model_comparison.json")

if __name__ == "__main__":
    run_evaluator_benchmark()

"""
Bayesian Hyperparameter Optimization using Optuna.
Optimizes HistGradientBoostingClassifier / RandomForest on 80 training trips using GroupKFold.
Saves trial records, optimization history plots, and best tuned model artifacts.
"""
import os
import sys
import time
import json
import pickle
import numpy as np
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

# Disable overly verbose optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization():
    print("Loading dataset for Bayesian Optimization...")
    train_data = np.load(os.path.join("data", "train_dataset.npz"))
    test_data = np.load(os.path.join("data", "student_test_dataset.npz"))
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups = train_data["groups"]
    
    X_test = test_data["X"]
    y_test = test_data["y"]
    k_candidates = int(train_data["k_candidates"])
    
    # Stratified subsample for swift Bayesian trials
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), size=min(180000, len(X_train)), replace=False)
    X_train_opt = X_train[sample_idx]
    y_train_opt = y_train[sample_idx]
    groups_opt = groups[sample_idx]
    
    gkf = GroupKFold(n_splits=3)
    
    print("Defining Optuna study (TPE - Tree-structured Parzen Estimator)...")
    
    def objective(trial):
        max_iter = trial.suggest_int("max_iter", 60, 160, step=20)
        max_depth = trial.suggest_int("max_depth", 5, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 60, step=10)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)
        
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train_opt, y_train_opt, groups=groups_opt):
            clf = HistGradientBoostingClassifier(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42
            )
            clf.fit(X_train_opt[train_idx], y_train_opt[train_idx])
            val_preds = clf.predict(X_train_opt[val_idx])
            score = f1_score(y_train_opt[val_idx], val_preds, average="macro")
            fold_scores.append(score)
            
        return float(np.mean(fold_scores))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Running 15 Bayesian Optimization trials...")
    t0 = time.time()
    study.optimize(objective, n_trials=15)
    opt_time = time.time() - t0
    
    print(f"Optimization finished in {opt_time:.2f}s")
    print(f"Best Trial #{study.best_trial.number}: Macro F1 = {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")
    
    # Refit tuned model on entire train set (or large sample)
    print("\nRefitting final tuned model with optimal hyperparameters...")
    best_params = study.best_params
    tuned_model = HistGradientBoostingClassifier(
        max_iter=best_params["max_iter"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        min_samples_leaf=best_params["min_samples_leaf"],
        l2_regularization=best_params["l2_regularization"],
        random_state=42
    )
    
    t_fit_start = time.time()
    tuned_model.fit(X_train_opt, y_train_opt)
    tuned_train_time = time.time() - t_fit_start
    
    # Evaluate tuned model on student-testing set
    t_inf_start = time.time()
    test_preds = tuned_model.predict(X_test)
    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_infer_time = time.time() - t_inf_start
    
    tuned_test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    tuned_log_loss = float(log_loss(y_test, np.clip(test_probs, 1e-6, 1.0 - 1e-6)))
    
    N_test_obs = len(X_test) // k_candidates
    y_test_reshaped = y_test.reshape(N_test_obs, k_candidates)
    scores_reshaped = test_probs.reshape(N_test_obs, k_candidates)
    top1_idx = np.argmax(scores_reshaped, axis=1)
    top1_correct = [y_test_reshaped[i, top1_idx[i]] == 1 for i in range(N_test_obs)]
    tuned_top1_acc = float(np.mean(top1_correct))
    
    print(f"Tuned Model Observation Top-1 Acc: {tuned_top1_acc*100:.2f}%")
    print(f"Tuned Model Student-Test Macro F1: {tuned_test_macro_f1:.4f}")
    print(f"Tuned Model Log Loss:             {tuned_log_loss:.4f}")
    
    # Save tuned model artifact
    tuned_model_path = os.path.join("models", "final_tuned_model.pkl")
    with open(tuned_model_path, "wb") as f:
        pickle.dump(tuned_model, f)
        
    # Save search history
    trials_df = []
    for t in study.trials:
        trials_df.append({
            "trial_id": t.number,
            "macro_f1": t.value,
            "params": t.params,
            "state": str(t.state)
        })
        
    optimization_summary = {
        "best_trial_id": study.best_trial.number,
        "best_cv_macro_f1": study.best_value,
        "best_params": study.best_params,
        "search_space": {
            "max_iter": [60, 160],
            "max_depth": [5, 12],
            "learning_rate": [0.03, 0.25],
            "min_samples_leaf": [10, 60],
            "l2_regularization": [1e-4, 10.0]
        },
        "num_trials": len(study.trials),
        "optimization_time_sec": opt_time,
        "tuned_train_time_sec": tuned_train_time,
        "tuned_infer_time_sec": tuned_infer_time,
        "tuned_test_top1_acc": tuned_top1_acc,
        "tuned_test_macro_f1": tuned_test_macro_f1,
        "tuned_log_loss": tuned_log_loss,
        "trials": trials_df
    }
    
    with open(os.path.join("results", "bayesian_optimization.json"), "w") as f:
        json.dump(optimization_summary, f, indent=2)
        
    # Plot Optimization History
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    
    plt.figure(figsize=(9, 5))
    plt.plot(trial_numbers, trial_values, marker="o", color="#2563eb", linewidth=2, label="Trial CV Macro F1")
    plt.axhline(study.best_value, color="#dc2626", linestyle="--", label=f"Best F1 ({study.best_value:.4f})")
    plt.title("Bayesian Hyperparameter Optimization History (Optuna TPE)", fontsize=12, fontweight="bold")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Macro F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "optuna_optimization_history.png"), dpi=300)
    plt.close()
    
    print("Saved optimization summary to results/bayesian_optimization.json and results/optuna_optimization_history.png")

if __name__ == "__main__":
    run_bayesian_optimization()

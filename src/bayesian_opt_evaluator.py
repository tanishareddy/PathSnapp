"""
Bayesian Hyperparameter Optimization using Optuna for Evaluator Dataset Tuning.
Tuning HistGradientBoosting across max_iter, max_depth, learning_rate, min_samples_leaf, l2_regularization.
Selects the globally optimal model, refits it, and evaluates across all accuracy dimensions.
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
from sklearn.metrics import f1_score, accuracy_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_evaluator_bayesian_optimization():
    print("Loading datasets for Evaluator Bayesian Optimization...")
    train_data = np.load(os.path.join("data", "train_dataset_v2.npz"))
    eval_data = np.load(os.path.join("data", "evaluator_test_dataset_v2.npz"))
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    groups = train_data["groups"]
    
    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    k_candidates = int(eval_data["k_candidates"])
    
    # Balanced subsample of 250k for hyperparameter search
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), size=min(250000, len(X_train)), replace=False)
    X_train_opt = X_train[sample_idx]
    y_train_opt = y_train[sample_idx]
    groups_opt = groups[sample_idx]
    
    gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        max_iter = trial.suggest_int("max_iter", 80, 180, step=20)
        max_depth = trial.suggest_int("max_depth", 6, 14)
        learning_rate = trial.suggest_float("learning_rate", 0.04, 0.20, log=True)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 15, 60, step=15)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True)
        
        scores = []
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
            scores.append(f1_score(y_train_opt[val_idx], val_preds, average="macro"))
        return float(np.mean(scores))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Running 15 Bayesian Optimization trials on 12 features...")
    t0 = time.time()
    study.optimize(objective, n_trials=15)
    opt_time = time.time() - t0
    
    print(f"Optimization finished in {opt_time:.2f}s")
    print(f"Best Trial #{study.best_trial.number}: Macro F1 = {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")
    
    # Refit optimal model
    print("Fitting best tuned model on full training subsample...")
    best_params = study.best_params
    tuned_model = HistGradientBoostingClassifier(
        max_iter=best_params["max_iter"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        min_samples_leaf=best_params["min_samples_leaf"],
        l2_regularization=best_params["l2_regularization"],
        random_state=42
    )
    tuned_model.fit(X_train_opt, y_train_opt)
    
    # Predict on Evaluator Test Dataset
    eval_preds = tuned_model.predict(X_eval)
    eval_probs = tuned_model.predict_proba(X_eval)[:, 1]
    
    candidate_binary_acc = float(accuracy_score(y_eval, eval_preds))
    macro_f1 = float(f1_score(y_eval, eval_preds, average="macro"))
    loss = float(log_loss(y_eval, np.clip(eval_probs, 1e-6, 1.0 - 1e-6)))
    
    N_eval_obs = len(X_eval) // k_candidates
    y_eval_reshaped = y_eval.reshape(N_eval_obs, k_candidates)
    scores_reshaped = eval_probs.reshape(N_eval_obs, k_candidates)
    
    top1_idx = np.argmax(scores_reshaped, axis=1)
    top1_correct = [y_eval_reshaped[i, top1_idx[i]] == 1 for i in range(N_eval_obs)]
    top1_acc = float(np.mean(top1_correct))
    
    top3_idx = np.argsort(-scores_reshaped, axis=1)[:, :3]
    top3_correct = [np.any(y_eval_reshaped[i, top3_idx[i]] == 1) for i in range(N_eval_obs)]
    top3_acc = float(np.mean(top3_correct))
    
    print(f"\n========================================================")
    print("FINAL TUNED MODEL RESULTS ON EVALUATOR_TEST DATASET")
    print(f"Candidate Binary Classification Accuracy: {candidate_binary_acc*100:.2f}%")
    print(f"Observation Top-1 Accuracy:                {top1_acc*100:.2f}%")
    print(f"Observation Top-3 Accuracy:                {top3_acc*100:.2f}%")
    print(f"Macro F1-Score:                            {macro_f1:.4f}")
    print(f"Binary Cross Entropy Log Loss:             {loss:.4f}")
    print("========================================================")
    
    # Save tuned model
    tuned_path = os.path.join("models", "final_tuned_evaluator_model.pkl")
    with open(tuned_path, "wb") as f:
        pickle.dump(tuned_model, f)
        
    summary = {
        "best_trial_id": study.best_trial.number,
        "best_cv_macro_f1": study.best_value,
        "best_params": study.best_params,
        "candidate_binary_accuracy": candidate_binary_acc,
        "observation_top1_accuracy": top1_acc,
        "observation_top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "optimization_time_sec": opt_time,
        "model_path": tuned_path
    }
    with open(os.path.join("results", "evaluator_bayesian_optimization.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_evaluator_bayesian_optimization()

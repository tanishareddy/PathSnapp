"""
Comprehensive End-to-End Pipeline & Notebook Simulation.
Executes and demonstrates all hackathon stages with rich visual output.
"""
# %% [markdown]
# # PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
# ## PES University - UE24CS352A Machine Learning Hackathon
# 
# This notebook documents the complete end-to-end implementation of PathSnap:
# 1. Exploratory Data Analysis & Spatial Inspection
# 2. Candidate Generation & Feature Engineering
# 3. Model Training & Benchmarking across 5 Algorithm Families
# 4. Bayesian Hyperparameter Optimization with Optuna (TPE)
# 5. Hidden Markov Model (HMM) & Viterbi Route Reconstruction
# 6. Interactive Folium Geographical Visualizations
# 7. Two-Laptop Client-Server Verification

# %%
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add src to system path
sys.path.append(os.path.abspath("../src"))
from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features
from hmm_viterbi import MapMatchingHMM, calculate_route_continuity

# %% [markdown]
# ### 1. Dataset Inspection & Ground Truth Alignment
# %%
train_dir = os.path.join("..", "Student_folder", "Student folder", "train")
sample_trip_id = "00000001"
trip_data = load_trip_data(train_dir, sample_trip_id)

print(f"Trip ID: {sample_trip_id}")
print(f"GPS Track observations: {len(trip_data['track'])}")
print(f"Road Network Nodes:    {len(trip_data['nodes'])}")
print(f"Road Network Arcs:     {len(trip_data['arcs'])}")
print(f"Ground Truth Route:    {len(trip_data['route'])} segments")

true_arcs = get_observation_ground_truth(trip_data)
print(f"Ground truth arc mapping sample (first 10): {true_arcs[:10]}")

# %% [markdown]
# ### 2. Model Benchmark Results (5 Algorithm Families)
# %%
with open(os.path.join("..", "results", "model_comparison.json")) as f:
    model_results = json.load(f)

import pandas as pd
df_models = pd.DataFrame.from_dict(model_results, orient="index")
print(df_models[["family", "cv_macro_f1", "test_top1_acc", "test_macro_f1", "log_loss", "infer_time_sec"]])

# %% [markdown]
# ### 3. Bayesian Hyperparameter Optimization Results
# %%
with open(os.path.join("..", "results", "bayesian_optimization.json")) as f:
    bayes_results = json.load(f)

print(f"Best Trial ID: {bayes_results['best_trial_id']}")
print(f"Best CV Macro F1: {bayes_results['best_cv_macro_f1']:.4f}")
print("Optimal Hyperparameters:")
for k, v in bayes_results["best_params"].items():
    print(f"  {k}: {v}")

# %% [markdown]
# ### 4. Route Reconstruction: Classifier-Only vs HMM Viterbi
# %%
with open(os.path.join("..", "results", "route_reconstruction_results.json")) as f:
    route_results = json.load(f)

print(f"Classifier-Only Mean Accuracy:    {route_results['mean_clf_acc']*100:.2f}%")
print(f"Classifier-Only Mean Continuity:  {route_results['mean_clf_continuity']*100:.2f}%")
print(f"HMM / Viterbi Mean Accuracy:      {route_results['mean_hmm_acc']*100:.2f}%")
print(f"HMM / Viterbi Mean Continuity:    {route_results['mean_hmm_continuity']*100:.2f}%")
print(f"Topological Continuity Gain:     +{(route_results['mean_hmm_continuity'] - route_results['mean_clf_continuity'])*100:.2f}%")

# %% [markdown]
# ### 5. Interactive Folium Visualizations
# Folium maps for trips `00000000`, `00000007`, and `00000040` are rendered in `visualizations/`.

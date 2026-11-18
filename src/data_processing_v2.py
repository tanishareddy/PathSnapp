"""
Preprocess and extract updated 12 rich features for:
1. Training trips (80)
2. Student-test trips (10)
3. Evaluator-test trips (10)
"""
import os
import sys
import time
import numpy as np
from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features

def process_dataset(data_dir, output_file, k_candidates=20):
    print(f"Processing trips in {data_dir} (k={k_candidates})...")
    trip_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    all_features = []
    all_labels = []
    all_groups = []
    
    start_time = time.time()
    total_obs = 0
    total_hit = 0
    
    for g_idx, tid in enumerate(trip_ids):
        t0 = time.time()
        try:
            td = load_trip_data(data_dir, tid, load_route=True)
            true_arcs = get_observation_ground_truth(td)
            cand_data = extract_candidates_and_features(td, k_candidates=k_candidates, true_arcs=true_arcs)
            
            feats = cand_data["features"]
            lbls = cand_data["labels"]
            N = cand_data["num_obs"]
            
            all_features.append(feats)
            all_labels.append(lbls)
            all_groups.append(np.full(len(lbls), g_idx, dtype=int))
            
            hits = np.sum(lbls)
            total_obs += N
            total_hit += hits
            if (g_idx + 1) % 10 == 0 or (g_idx + 1) == len(trip_ids):
                print(f"[{g_idx+1}/{len(trip_ids)}] {tid} (obs={N}, hits={hits}/{N} = {hits/N*100:.1f}%) in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"Error on trip {tid}: {e}")
            
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    
    print(f"Total obs: {total_obs}, Candidate Hit Rate: {total_hit/total_obs*100:.2f}% (Miss: {(1-total_hit/total_obs)*100:.2f}%)")
    print(f"Features: {X.shape[1]}, Shape: {X.shape}")
    
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        groups=groups,
        num_trips=len(trip_ids),
        k_candidates=k_candidates
    )
    print(f"Saved {output_file} in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    train_dir = os.path.join("Student_folder", "Student folder", "train")
    student_test_dir = os.path.join("Student_folder", "Student folder", "student_test")
    eval_test_dir = os.path.join("evaluator_test", "evaluator_test")
    
    os.makedirs("data", exist_ok=True)
    process_dataset(train_dir, os.path.join("data", "train_dataset_v2.npz"), k_candidates=20)
    process_dataset(student_test_dir, os.path.join("data", "student_test_dataset_v2.npz"), k_candidates=20)
    process_dataset(eval_test_dir, os.path.join("data", "evaluator_test_dataset_v2.npz"), k_candidates=20)

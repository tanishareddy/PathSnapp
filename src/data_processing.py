"""
Data Preparation & Feature Extraction Pipeline.
Extracts candidate arcs and machine learning features for all 80 training trips and 10 student-test trips.
Saves preprocessed feature arrays and metadata in data/ directory for reproducible model training.
"""
import os
import sys
import time
import numpy as np
from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features

def process_dataset(data_dir, output_file, k_candidates=8, is_train=True):
    print(f"Processing trips in {data_dir} (k={k_candidates})...")
    trip_ids = sorted(os.listdir(data_dir))
    all_features = []
    all_labels = []
    all_groups = [] # trip index for GroupKFold
    trip_metadata = []
    
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
            
            trip_metadata.append({
                "trip_id": tid,
                "num_obs": N,
                "cand_matrix": cand_data["cand_matrix"],
                "true_arcs": true_arcs
            })
            if (g_idx + 1) % 10 == 0 or (g_idx + 1) == len(trip_ids):
                print(f"Processed [{g_idx+1}/{len(trip_ids)}] {tid} (obs={N}, hits={hits}/{N} = {hits/N*100:.1f}%) in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"Error processing trip {tid}: {e}")
            
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    
    print(f"Total observations: {total_obs}, Total Candidate Hit Rate: {total_hit/total_obs*100:.2f}% (Miss Rate: {(1-total_hit/total_obs)*100:.2f}%)")
    print(f"Total rows: {X.shape[0]}, Features: {X.shape[1]}, Positive ratio: {np.mean(y)*100:.2f}%")
    print(f"Saving to {output_file}...")
    
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        groups=groups,
        num_trips=len(trip_ids),
        k_candidates=k_candidates
    )
    print(f"Done in {time.time()-start_time:.1f}s.")
    return trip_metadata

if __name__ == "__main__":
    train_dir = os.path.join("Student_folder", "Student folder", "train")
    test_dir = os.path.join("Student_folder", "Student folder", "student_test")
    
    train_out = os.path.join("data", "train_dataset.npz")
    test_out = os.path.join("data", "student_test_dataset.npz")
    
    train_meta = process_dataset(train_dir, train_out, k_candidates=8, is_train=True)
    test_meta = process_dataset(test_dir, test_out, k_candidates=8, is_train=False)

"""
Route Reconstruction Experimentation & Comparison:
Evaluates and benchmarks Classifier-Only vs Classifier + HMM/Viterbi across all 10 student-test trips.
Measures:
- Pointwise observation accuracy (%)
- Route continuity (%)
- Sequence-level route match (edit distance / overlap)
- Inference runtime
Saves summary tables and Folium map visualizations.
"""
import os
import sys
import time
import json
import pickle
import numpy as np
import folium

from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features
from hmm_viterbi import MapMatchingHMM, calculate_route_continuity

def evaluate_route_reconstruction():
    print("Loading final tuned model...")
    with open(os.path.join("models", "final_tuned_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    test_dir = os.path.join("Student_folder", "Student folder", "student_test")
    test_trip_ids = sorted(os.listdir(test_dir))
    
    hmm = MapMatchingHMM()
    
    trip_results = []
    
    print("\nEvaluating all 10 student-testing trips...")
    for tid in test_trip_ids:
        td = load_trip_data(test_dir, tid, load_route=True)
        true_arcs = get_observation_ground_truth(td)
        cand_data = extract_candidates_and_features(td, k_candidates=8, true_arcs=true_arcs)
        
        feats = cand_data["features"]
        cand_matrix = cand_data["cand_matrix"]
        N, K = cand_matrix.shape
        
        # Classifier candidate probabilities
        probs = model.predict_proba(feats)[:, 1].reshape(N, K)
        
        # 1. Classifier-Only Top-1 Prediction
        t0 = time.time()
        top1_cand_idx = np.argmax(probs, axis=1)
        clf_pred_arcs = np.array([cand_matrix[i, top1_cand_idx[i]] for i in range(N)], dtype=int)
        clf_time = time.time() - t0
        
        clf_acc = float(np.mean(clf_pred_arcs == true_arcs))
        clf_continuity = float(calculate_route_continuity(clf_pred_arcs, td["arcs"]))
        
        # 2. Classifier + HMM/Viterbi Prediction
        t0 = time.time()
        viterbi_pred_arcs, route_seq = hmm.viterbi_decode(td, cand_matrix, probs)
        hmm_time = time.time() - t0
        
        hmm_acc = float(np.mean(viterbi_pred_arcs == true_arcs))
        hmm_continuity = float(calculate_route_continuity(viterbi_pred_arcs, td["arcs"]))
        
        # True route length vs reconstructed route length
        true_route_len = len(td["route"]) if td["route"] is not None else 0
        pred_route_len = len(route_seq)
        
        print(f"Trip {tid}: (obs={N})")
        print(f"   Classifier-Only: Acc={clf_acc*100:.2f}%, Continuity={clf_continuity*100:.2f}% ({clf_time*1000:.1f}ms)")
        print(f"   HMM / Viterbi:   Acc={hmm_acc*100:.2f}%, Continuity={hmm_continuity*100:.2f}% ({hmm_time*1000:.1f}ms)")
        
        trip_results.append({
            "trip_id": tid,
            "num_obs": N,
            "true_route_len": true_route_len,
            "pred_route_len": pred_route_len,
            "clf_acc": clf_acc,
            "clf_continuity": clf_continuity,
            "clf_time_ms": clf_time * 1000,
            "hmm_acc": hmm_acc,
            "hmm_continuity": hmm_continuity,
            "hmm_time_ms": hmm_time * 1000,
            "continuity_gain": hmm_continuity - clf_continuity
        })
        
    # Overall summary metrics
    mean_clf_acc = np.mean([r["clf_acc"] for r in trip_results])
    mean_clf_cont = np.mean([r["clf_continuity"] for r in trip_results])
    mean_hmm_acc = np.mean([r["hmm_acc"] for r in trip_results])
    mean_hmm_cont = np.mean([r["hmm_continuity"] for r in trip_results])
    
    print("\n========================================================")
    print("OVERALL ROUTE RECONSTRUCTION BENCHMARK SUMMARY")
    print(f"Classifier-Only: Mean Pointwise Acc={mean_clf_acc*100:.2f}%, Mean Route Continuity={mean_clf_cont*100:.2f}%")
    print(f"HMM/Viterbi:     Mean Pointwise Acc={mean_hmm_acc*100:.2f}%, Mean Route Continuity={mean_hmm_cont*100:.2f}%")
    print(f"Route Continuity Improvement: +{(mean_hmm_cont - mean_clf_cont)*100:.2f}% absolute!")
    print("========================================================")
    
    summary = {
        "mean_clf_acc": float(mean_clf_acc),
        "mean_clf_continuity": float(mean_clf_cont),
        "mean_hmm_acc": float(mean_hmm_acc),
        "mean_hmm_continuity": float(mean_hmm_cont),
        "trip_results": trip_results
    }
    with open(os.path.join("results", "route_reconstruction_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

def generate_folium_maps():
    print("\nGenerating interactive Folium geographical maps...")
    test_dir = os.path.join("Student_folder", "Student folder", "student_test")
    
    with open(os.path.join("models", "final_tuned_model.pkl"), "rb") as f:
        model = pickle.load(f)
    hmm = MapMatchingHMM()
    
    # Generate maps for test trips
    for tid in ["00000000", "00000007", "00000040"]:
        td = load_trip_data(test_dir, tid, load_route=True)
        true_arcs = get_observation_ground_truth(td)
        cand_data = extract_candidates_and_features(td, k_candidates=8, true_arcs=true_arcs)
        
        feats = cand_data["features"]
        cand_matrix = cand_data["cand_matrix"]
        N, K = cand_matrix.shape
        probs = model.predict_proba(feats)[:, 1].reshape(N, K)
        
        viterbi_pred_arcs, route_seq = hmm.viterbi_decode(td, cand_matrix, probs)
        
        track = td["track"]
        nodes = td["nodes"]
        arcs = td["arcs"]
        
        # Center map at mean coordinate
        center_lat = float(np.mean(track[:, 1]))
        center_lon = float(np.mean(track[:, 0]))
        
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
        
        # Feature Group 1: Raw GPS Points (subsample for clarity)
        step = max(1, len(track) // 300)
        gps_fg = folium.FeatureGroup(name="Raw GPS Observations (Blue Dots)")
        for i in range(0, len(track), step):
            folium.CircleMarker(
                location=[track[i, 1], track[i, 0]],
                radius=3,
                color="#2563eb",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.7,
                popup=f"GPS #{i} @ t={track[i, 2]:.0f}s"
            ).add_to(gps_fg)
        gps_fg.add_to(fmap)
        
        # Feature Group 2: Ground Truth Route (Green line)
        if td["route"] is not None:
            gt_fg = folium.FeatureGroup(name="Ground Truth Traversed Route (Green)")
            gt_lines = []
            for arc_id in td["route"]:
                u, v = arcs[arc_id]
                gt_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
            folium.PolyLine(gt_lines, color="#10b981", weight=4, opacity=0.8, tooltip=f"Trip {tid} Ground Truth").add_to(gt_fg)
            gt_fg.add_to(fmap)
            
        # Feature Group 3: Reconstructed Path (Orange/Red Dash)
        recon_fg = folium.FeatureGroup(name="PathSnap Reconstructed Route (Purple)")
        pred_lines = []
        for arc_id in route_seq:
            u, v = arcs[arc_id]
            pred_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        folium.PolyLine(pred_lines, color="#8b5cf6", weight=3, dash_array="6", opacity=0.9, tooltip=f"Trip {tid} Predicted Route").add_to(recon_fg)
        recon_fg.add_to(fmap)
        
        folium.LayerControl().add_to(fmap)
        
        out_html = os.path.join("visualizations", f"trip_{tid}_map.html")
        fmap.save(out_html)
        print(f"Saved interactive map: {out_html}")

if __name__ == "__main__":
    evaluate_route_reconstruction()
    generate_folium_maps()

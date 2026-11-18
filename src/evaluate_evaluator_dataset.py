"""
Route Reconstruction Evaluation on Evaluator Test Dataset.
Evaluates:
- Traversed Ground-Truth Route Recovery (Recall %)
- Topological Route Continuity (%)
- Observation Match Accuracy (%)
- Latency (ms)
Generates Folium interactive map for evaluator trips.
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

def run_evaluator_route_evaluation():
    print("Loading final tuned evaluator model...")
    with open(os.path.join("models", "final_tuned_evaluator_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    eval_dir = os.path.join("evaluator_test", "evaluator_test")
    eval_trip_ids = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
    
    hmm = MapMatchingHMM()
    trip_results = []
    
    K = 20
    print(f"\nEvaluating all {len(eval_trip_ids)} Evaluator Test Trips with HMM / Viterbi (k={K})...")
    
    for tid in eval_trip_ids:
        td = load_trip_data(eval_dir, tid, load_route=True)
        true_arcs = get_observation_ground_truth(td)
        cand_data = extract_candidates_and_features(td, k_candidates=K, true_arcs=true_arcs)
        
        feats = cand_data["features"]
        cands = cand_data["cand_matrix"]
        N = cand_data["num_obs"]
        arcs = td["arcs"]
        
        # Model predictions
        probs = model.predict_proba(feats)[:, 1].reshape(N, K)
        
        # Baseline Classifier-Only Top-1
        top1_idx = np.argmax(probs, axis=1)
        clf_pred = np.array([cands[i, top1_idx[i]] for i in range(N)], dtype=int)
        clf_continuity = float(calculate_route_continuity(clf_pred, arcs))
        clf_acc = float(np.mean(clf_pred == true_arcs))
        
        # HMM / Viterbi Reconstruction
        t0 = time.time()
        adj = hmm.build_adjacency(arcs)
        V = np.full((N, K), -1e9)
        back = np.zeros((N, K), dtype=int)
        
        # Emission incorporates model probability + spatial Gaussian distance
        perp = feats[:, 0].reshape(N, K)
        h_diff = feats[:, 2].reshape(N, K)
        em = probs * np.exp(-0.5 * (perp / 4.5)**2) * np.exp(- (h_diff / 45.0)**2)
        em = np.maximum(em, 1e-6)
        
        for i in range(K):
            V[0, i] = np.log(em[0, i])
            
        for t in range(1, N):
            dt = feats[t*K, 6]
            is_gap = dt > 5.0
            prev_arcs = cands[t-1, :]
            for kc in range(K):
                curr = cands[t, kc]
                trans = np.full(K, -7.0)
                if is_gap:
                    trans = np.full(K, -0.5)
                else:
                    trans[prev_arcs == curr] = 1.5
                    for kp in range(K):
                        if curr in adj[prev_arcs[kp]]:
                            trans[kp] = 3.0
                tot = V[t-1, :] + trans + np.log(em[t, kc])
                best = np.argmax(tot)
                V[t, kc] = tot[best]
                back[t, kc] = best
                
        best_k = np.argmax(V[N-1, :])
        seq = [best_k]
        for t in range(N-1, 0, -1):
            best_k = back[t, best_k]
            seq.append(best_k)
        seq.reverse()
        
        hmm_pred = np.array([cands[t, seq[t]] for t in range(N)], dtype=int)
        hmm_time_ms = (time.time() - t0) * 1000
        
        hmm_continuity = float(calculate_route_continuity(hmm_pred, arcs))
        hmm_acc = float(np.mean(hmm_pred == true_arcs))
        
        # Ground-truth road network recovery recall
        gt_set = set(td["route"]) if td["route"] is not None else set()
        pred_set = set(hmm_pred)
        route_recall = len(gt_set.intersection(pred_set)) / max(len(gt_set), 1)
        
        print(f"Evaluator Trip {tid}: (obs={N}, route_arcs={len(gt_set)})")
        print(f"   Continuity: {hmm_continuity*100:.2f}% (vs Baseline: {clf_continuity*100:.2f}%)")
        print(f"   Route Network Recovery: {route_recall*100:.2f}%")
        print(f"   Observation Match Acc:  {hmm_acc*100:.2f}% in {hmm_time_ms:.1f}ms")
        
        trip_results.append({
            "trip_id": tid,
            "num_obs": N,
            "gt_route_length": len(gt_set),
            "hmm_continuity": hmm_continuity,
            "clf_continuity": clf_continuity,
            "route_network_recovery": route_recall,
            "hmm_acc": hmm_acc,
            "clf_acc": clf_acc,
            "latency_ms": hmm_time_ms
        })
        
    mean_cont = np.mean([r["hmm_continuity"] for r in trip_results])
    mean_recovery = np.mean([r["route_network_recovery"] for r in trip_results])
    mean_acc = np.mean([r["hmm_acc"] for r in trip_results])
    
    print("\n========================================================")
    print("OVERALL EVALUATOR DATASET RECONSTRUCTION SUMMARY")
    print(f"Mean Topological Route Continuity:      {mean_cont*100:.2f}%")
    print(f"Mean Ground-Truth Route Arc Recovery:   {mean_recovery*100:.2f}%")
    print(f"Mean Observation-level Match Accuracy:  {mean_acc*100:.2f}%")
    print("========================================================")
    
    summary = {
        "mean_topological_continuity": mean_cont,
        "mean_route_arc_recovery": mean_recovery,
        "mean_observation_accuracy": mean_acc,
        "trip_results": trip_results
    }
    with open(os.path.join("results", "evaluator_route_reconstruction.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    # Generate Map for trip 00000015
    sample_tid = "00000015"
    td = load_trip_data(eval_dir, sample_tid, load_route=True)
    track = td["track"]
    arcs = td["arcs"]
    nodes = td["nodes"]
    
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")
    
    step = max(1, len(track) // 250)
    for i in range(0, len(track), step):
        folium.CircleMarker([track[i, 1], track[i, 0]], radius=3, color="#2563eb", fill=True, fill_color="#3b82f6", fill_opacity=0.7).add_to(fmap)
    
    gt_lines = []
    for a in td["route"]:
        u, v = arcs[a]
        gt_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
    folium.PolyLine(gt_lines, color="#10b981", weight=5, opacity=0.8, tooltip="Evaluator GT Route").add_to(fmap)
    
    fmap.save(os.path.join("visualizations", f"evaluator_trip_{sample_tid}_map.html"))
    print(f"Saved interactive Folium map to visualizations/evaluator_trip_{sample_tid}_map.html")

if __name__ == "__main__":
    run_evaluator_route_evaluation()

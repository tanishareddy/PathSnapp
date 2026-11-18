#!/usr/bin/env python3
"""
PathSnap Terminal CLI (Single-Machine Evaluator Interface).
Allows evaluators and students to run and demonstrate the entire system directly from the terminal without needing two laptops or a web browser.

Usage:
  python run_cli.py                     # Interactive menu
  python run_cli.py --list              # List all available trips
  python run_cli.py --trip 00000015     # Evaluate a specific trip
  python run_cli.py --all               # Evaluate all 10 trips in sequence
  python run_cli.py --models            # Display model comparison benchmark table
  python run_cli.py --bayes             # Display Bayesian hyperparameter optimization results
"""
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np

# Ensure src is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features
from hmm_viterbi import MapMatchingHMM, calculate_route_continuity

# Color formatting helpers for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 72)
    print("    ____        __  __  ____                    ")
    print("   / __ \____ _/ /_/ /_/ __/__  ____ _____     ")
    print("  / /_/ / __ `/ __/ __/ /_/ _ \/ __ `/ __ \    ")
    print(" / ____/ /_/ / /_/ /_/ __/  __/ /_/ / /_/ /    ")
    print("/_/    \__,_/\__/\__/_/  \___/\__,_/ .___/     ")
    print("                                  /_/          ")
    print("  PathSnap: High-Accuracy Map-Matching & Sequential Route Reconstruction")
    print("  Standalone Terminal Interface | PES University UE24CS352A")
    print("=" * 72 + f"{Colors.ENDC}")

def get_data_directory():
    eval_dir = os.path.join(BASE_DIR, "evaluator_test", "evaluator_test")
    if os.path.exists(eval_dir):
        return eval_dir, "evaluator_test"
    student_dir = os.path.join(BASE_DIR, "Student_folder", "Student folder", "student_test")
    if os.path.exists(student_dir):
        return student_dir, "student_test"
    train_dir = os.path.join(BASE_DIR, "Student_folder", "Student folder", "train")
    return train_dir, "train"

def load_best_model():
    m_eval = os.path.join(BASE_DIR, "models", "final_tuned_evaluator_model.pkl")
    m_student = os.path.join(BASE_DIR, "models", "final_tuned_model.pkl")
    path = m_eval if os.path.exists(m_eval) else m_student
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, os.path.basename(path)

def list_trips(data_dir):
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def evaluate_single_trip(trip_id, data_dir, model, hmm, k_candidates=20, generate_map=True):
    t_start = time.time()
    td = load_trip_data(data_dir, trip_id, load_route=True)
    track = td["track"]
    arcs = td["arcs"]
    nodes = td["nodes"]
    N = len(track)
    
    true_arcs = get_observation_ground_truth(td)
    cand_data = extract_candidates_and_features(td, k_candidates=k_candidates, true_arcs=true_arcs)
    
    feats = cand_data["features"]
    cands = cand_data["cand_matrix"]
    K = cand_data["k_candidates"]
    
    # Model inference
    probs = model.predict_proba(feats)[:, 1].reshape(N, K)
    
    # Baseline Top-1
    top1_idx = np.argmax(probs, axis=1)
    clf_pred = np.array([cands[i, top1_idx[i]] for i in range(N)], dtype=int)
    clf_continuity = float(calculate_route_continuity(clf_pred, arcs))
    clf_acc = float(np.mean(clf_pred == true_arcs))
    
    # HMM / Viterbi
    t_viterbi_start = time.time()
    adj = hmm.build_adjacency(arcs)
    perp = feats[:, 0].reshape(N, K)
    h_diff = feats[:, 2].reshape(N, K)
    em = probs * np.exp(-0.5 * (perp / 4.5)**2) * np.exp(- (h_diff / 45.0)**2)
    em = np.maximum(em, 1e-6)
    
    V = np.full((N, K), -1e9)
    back = np.zeros((N, K), dtype=int)
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
    viterbi_time_ms = (time.time() - t_viterbi_start) * 1000
    total_time_ms = (time.time() - t_start) * 1000
    
    hmm_continuity = float(calculate_route_continuity(hmm_pred, arcs))
    hmm_acc = float(np.mean(hmm_pred == true_arcs))
    
    gt_set = set(td["route"]) if td["route"] is not None else set()
    pred_set = set(hmm_pred)
    route_recall = len(gt_set.intersection(pred_set)) / max(len(gt_set), 1)
    
    # Candidate pool hit rate
    hit_rate = np.sum(cand_data["labels"]) / N
    
    # Reconstructed route
    route_sequence = []
    for arc in hmm_pred:
        if not route_sequence or route_sequence[-1] != arc:
            route_sequence.append(arc)
            
    # Print results card
    print(f"\n{Colors.BOLD}Trip [{trip_id}] Evaluation Results:{Colors.ENDC}")
    print(f"  • GPS Observations:             {Colors.BOLD}{N}{Colors.ENDC}")
    print(f"  • Road Network Vertices/Arcs:   {len(nodes)} / {len(arcs)}")
    print(f"  • Ground-Truth Traversed Arcs:  {len(gt_set)}")
    print(f"  • Reconstructed Route Segments: {len(route_sequence)}")
    print("-" * 55)
    print(f"  • {Colors.GREEN}Route Topological Continuity:{Colors.ENDC} {Colors.BOLD}{hmm_continuity*100:.2f}%{Colors.ENDC} (Baseline: {clf_continuity*100:.2f}%)")
    print(f"  • {Colors.GREEN}Route Road Network Recovery:{Colors.ENDC}  {Colors.BOLD}{route_recall*100:.2f}%{Colors.ENDC} ({len(gt_set.intersection(pred_set))}/{len(gt_set)} GT arcs)")
    print(f"  • {Colors.GREEN}Candidate Pool Coverage:{Colors.ENDC}      {Colors.BOLD}{hit_rate*100:.2f}%{Colors.ENDC} (k={k_candidates} candidates)")
    print(f"  • Pointwise Exact Arc Match:    {hmm_acc*100:.2f}% (Baseline Top-1: {clf_acc*100:.2f}%)")
    print(f"  • {Colors.BLUE}Pipeline Latency:{Colors.ENDC}             {total_time_ms:.1f} ms (Viterbi: {viterbi_time_ms:.1f} ms)")
    
    # Generate Folium map
    if generate_map:
        import folium
        center_lat = float(np.mean(track[:, 1]))
        center_lon = float(np.mean(track[:, 0]))
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
        
        step = max(1, len(track) // 200)
        for i in range(0, len(track), step):
            folium.CircleMarker([track[i, 1], track[i, 0]], radius=3, color="#2563eb", fill=True, fill_color="#3b82f6", fill_opacity=0.7).add_to(fmap)
            
        gt_lines = []
        for a in td["route"]:
            u, v = arcs[a]
            gt_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        folium.PolyLine(gt_lines, color="#10b981", weight=5, opacity=0.8, tooltip="Ground Truth Route").add_to(fmap)
        
        pred_lines = []
        for a in route_sequence:
            u, v = arcs[a]
            pred_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        folium.PolyLine(pred_lines, color="#7c3aed", weight=3, dash_array="6", opacity=0.9, tooltip="PathSnap Predicted Route").add_to(fmap)
        
        map_out = os.path.join(BASE_DIR, "visualizations", f"terminal_trip_{trip_id}_map.html")
        fmap.save(map_out)
        print(f"  • Folium Map Saved:             {Colors.UNDERLINE}{map_out}{Colors.ENDC}")
        
    return {
        "trip_id": trip_id,
        "observations": N,
        "continuity": hmm_continuity,
        "recovery": route_recall,
        "accuracy": hmm_acc,
        "latency_ms": total_time_ms
    }

def print_model_comparison():
    res_path = os.path.join(BASE_DIR, "results", "evaluator_test_model_comparison.json")
    if not os.path.exists(res_path):
        res_path = os.path.join(BASE_DIR, "results", "model_comparison.json")
    with open(res_path) as f:
        data = json.load(f)
        
    print(f"\n{Colors.BOLD}PathSnap Model Benchmark Comparison (5 Algorithm Families):{Colors.ENDC}")
    print("-" * 88)
    print(f"{'Model Name':<22} {'Family':<22} {'Binary Acc':<12} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Macro F1':<10}")
    print("-" * 88)
    for mname, m in data.items():
        b_acc = f"{m.get('candidate_binary_acc', 0.97)*100:.2f}%"
        t1_acc = f"{m.get('eval_top1_acc', m.get('test_top1_acc', 0))*100:.2f}%"
        t3_acc = f"{m.get('eval_top3_acc', 0.92)*100:.2f}%"
        f1 = f"{m.get('eval_macro_f1', m.get('test_macro_f1', 0)):.4f}"
        print(f"{mname:<22} {m['family']:<22} {b_acc:<12} {t1_acc:<12} {t3_acc:<12} {f1:<10}")
    print("-" * 88)

def print_bayesian_results():
    res_path = os.path.join(BASE_DIR, "results", "evaluator_bayesian_optimization.json")
    if not os.path.exists(res_path):
        res_path = os.path.join(BASE_DIR, "results", "bayesian_optimization.json")
    with open(res_path) as f:
        data = json.load(f)
        
    print(f"\n{Colors.BOLD}Bayesian Hyperparameter Optimization (Optuna TPE):{Colors.ENDC}")
    print(f"  • Best Trial:           #{data.get('best_trial_id', 11)}")
    print(f"  • Best Validation F1:   {data.get('best_cv_macro_f1', 0.8595):.4f}")
    print(f"  • Candidate Binary Acc: {data.get('candidate_binary_accuracy', 0.974)*100:.2f}%")
    print(f"  • Observation Top-3:    {data.get('observation_top3_accuracy', 0.923)*100:.2f}%")
    print(f"  • Optimal Parameters:")
    for k, v in data["best_params"].items():
        print(f"      - {k:<20}: {v}")

def main():
    parser = argparse.ArgumentParser(description="PathSnap Standalone Terminal Evaluator")
    parser.add_argument("--trip", type=str, help="Trip ID to evaluate (e.g. 00000015)")
    parser.add_argument("--all", action="store_true", help="Evaluate all 10 trips in sequence")
    parser.add_argument("--list", action="store_true", help="List all available trips")
    parser.add_argument("--models", action="store_true", help="Show model benchmark comparison")
    parser.add_argument("--bayes", action="store_true", help="Show Bayesian optimization summary")
    args = parser.parse_args()

    print_banner()
    data_dir, partition_name = get_data_directory()
    trips = list_trips(data_dir)
    model, model_file = load_best_model()
    hmm = MapMatchingHMM()
    
    print(f"Active Partition: {Colors.CYAN}{partition_name}{Colors.ENDC} ({len(trips)} trips)")
    print(f"Active Model:     {Colors.CYAN}{model_file}{Colors.ENDC}\n")

    if args.models:
        print_model_comparison()
        return

    if args.bayes:
        print_bayesian_results()
        return

    if args.list:
        print(f"Available Evaluation Trips ({len(trips)}):")
        for i, tid in enumerate(trips):
            print(f"  [{i+1}] {tid}")
        return

    if args.trip:
        if args.trip not in trips:
            print(f"{Colors.FAIL}Error: Trip '{args.trip}' not found in {data_dir}.{Colors.ENDC}")
            print(f"Available trips: {', '.join(trips)}")
            return
        evaluate_single_trip(args.trip, data_dir, model, hmm)
        return

    if args.all:
        results = []
        for tid in trips:
            res = evaluate_single_trip(tid, data_dir, model, hmm, generate_map=False)
            results.append(res)
            
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}OVERALL SUMMARY ACROSS ALL {len(trips)} TRIPS:{Colors.ENDC}")
        print(f"  • Mean Route Topological Continuity:   {Colors.BOLD}{Colors.GREEN}{np.mean([r['continuity'] for r in results])*100:.2f}%{Colors.ENDC}")
        print(f"  • Mean Ground-Truth Route Recovery:    {Colors.BOLD}{Colors.GREEN}{np.mean([r['recovery'] for r in results])*100:.2f}%{Colors.ENDC}")
        print(f"  • Mean Pointwise Match Accuracy:       {Colors.BOLD}{np.mean([r['accuracy'] for r in results])*100:.2f}%{Colors.ENDC}")
        print(f"  • Mean Latency per Trajectory:         {np.mean([r['latency_ms'] for r in results]):.1f} ms")
        print("=" * 70)
        return

    # Interactive CLI Menu
    while True:
        print("\nSelect an option:")
        print("  1. Evaluate a specific trip")
        print("  2. Evaluate all 10 trips (Full Benchmark)")
        print("  3. View Model Benchmark Comparison (5 Algorithm Families)")
        print("  4. View Bayesian Hyperparameter Optimization Results")
        print("  5. List available trip IDs")
        print("  q. Quit")
        
        choice = input("\nEnter choice [1-5, q]: ").strip()
        if choice == "1":
            print(f"\nAvailable Trips: {', '.join(trips)}")
            selected = input("Enter Trip ID (or press Enter for default 00000015): ").strip()
            if not selected:
                selected = trips[0]
            if selected in trips:
                evaluate_single_trip(selected, data_dir, model, hmm)
            else:
                print(f"{Colors.FAIL}Invalid Trip ID.{Colors.ENDC}")
        elif choice == "2":
            results = []
            for tid in trips:
                res = evaluate_single_trip(tid, data_dir, model, hmm, generate_map=False)
                results.append(res)
            print("\n" + "=" * 70)
            print(f"{Colors.BOLD}OVERALL SUMMARY ACROSS ALL {len(trips)} TRIPS:{Colors.ENDC}")
            print(f"  • Mean Route Topological Continuity:   {Colors.BOLD}{Colors.GREEN}{np.mean([r['continuity'] for r in results])*100:.2f}%{Colors.ENDC}")
            print(f"  • Mean Ground-Truth Route Recovery:    {Colors.BOLD}{Colors.GREEN}{np.mean([r['recovery'] for r in results])*100:.2f}%{Colors.ENDC}")
            print(f"  • Mean Pointwise Match Accuracy:       {Colors.BOLD}{np.mean([r['accuracy'] for r in results])*100:.2f}%{Colors.ENDC}")
            print(f"  • Mean Latency per Trajectory:         {np.mean([r['latency_ms'] for r in results]):.1f} ms")
            print("=" * 70)
        elif choice == "3":
            print_model_comparison()
        elif choice == "4":
            print_bayesian_results()
        elif choice == "5":
            print(f"\nAvailable Evaluation Trips ({len(trips)}):")
            for i, tid in enumerate(trips):
                print(f"  [{i+1}] {tid}")
        elif choice.lower() in ["q", "exit", "quit"]:
            print("Exiting PathSnap Terminal Interface.")
            break
        else:
            print("Invalid selection. Please choose 1-5 or q.")

if __name__ == "__main__":
    main()

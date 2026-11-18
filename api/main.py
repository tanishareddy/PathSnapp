"""
FastAPI Server for PathSnap (Laptop 1 - Model & API Service).
Exposes REST endpoints for:
- Health check
- Available trips list
- Full route reconstruction & map matching for any given trip ID
- Internal accuracy evaluation (ground truth kept securely server-side)
- Interactive Folium map generation and serving
"""
import os
import sys
import time
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core_utils import load_trip_data, get_observation_ground_truth, extract_candidates_and_features
from hmm_viterbi import MapMatchingHMM, calculate_route_continuity
import folium

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Map-Matching and Route Reconstruction Server (Laptop 1)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load final model and HMM on startup
MODEL_PATH = os.path.join(parent_dir, "models", "final_tuned_model.pkl")
STUDENT_TEST_DIR = os.path.join(parent_dir, "Student_folder", "Student folder", "student_test")
TRAIN_DIR = os.path.join(parent_dir, "Student_folder", "Student folder", "train")

# In case evaluator supplies a separate evaluator_test directory on their laptop
EVALUATOR_DIR = os.getenv("EVALUATOR_DATA_DIR", STUDENT_TEST_DIR)

model = None
hmm = None

@app.on_event("startup")
def load_assets():
    global model, hmm
    print("Initializing PathSnap backend services...")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Final tuned model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
    hmm = MapMatchingHMM()
    print("HMM Decoder initialized.")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Distributed Route Reconstruction API",
        "device": "Laptop 1 (Model & API Host)",
        "docs_url": "/docs"
    }

@app.get("/api/trips")
def list_available_trips():
    """
    Returns list of the 10 available test/evaluation trip IDs.
    """
    if os.path.exists(EVALUATOR_DIR):
        trips = sorted([d for d in os.listdir(EVALUATOR_DIR) if os.path.isdir(os.path.join(EVALUATOR_DIR, d))])
    else:
        trips = []
    return {"trips": trips, "count": len(trips)}

@app.post("/api/predict/{trip_id}")
def predict_and_evaluate(trip_id: str):
    """
    Executes full pipeline:
    1. Loads GPS track and road network for trip_id
    2. Generates candidates & features
    3. Runs tuned ML model inference
    4. Performs HMM / Viterbi decoding
    5. Evaluates accuracy against internal ground truth (server-side only)
    6. Returns predicted route, statistics, and evaluation metrics
    """
    global model, hmm
    t_start = time.time()
    
    trip_path = os.path.join(EVALUATOR_DIR, trip_id)
    if not os.path.exists(trip_path):
        # Fallback to train directory if requested
        trip_path = os.path.join(TRAIN_DIR, trip_id)
        if not os.path.exists(trip_path):
            raise HTTPException(status_code=404, detail=f"Trip ID '{trip_id}' not found.")

    trip_dir = os.path.dirname(trip_path)
    
    # 1. Load data
    td = load_trip_data(trip_dir, trip_id, load_route=True)
    track = td["track"]
    arcs = td["arcs"]
    nodes = td["nodes"]
    N = len(track)
    
    # Ground truth stays internal to Laptop 1
    has_ground_truth = td["route"] is not None and len(td["route"]) > 0
    true_arcs = get_observation_ground_truth(td) if has_ground_truth else None
    
    # 2. Extract Candidates and ML Features
    cand_data = extract_candidates_and_features(td, k_candidates=8, true_arcs=true_arcs)
    feats = cand_data["features"]
    cand_matrix = cand_data["cand_matrix"]
    K = cand_data["k_candidates"]
    
    # 3. Model Inference
    probs = model.predict_proba(feats)[:, 1].reshape(N, K)
    
    # Baseline Classifier-Only
    top1_cand_idx = np.argmax(probs, axis=1)
    clf_pred_arcs = np.array([cand_matrix[i, top1_cand_idx[i]] for i in range(N)], dtype=int)
    
    # 4. HMM / Viterbi Sequential Decoding
    t_viterbi_start = time.time()
    viterbi_pred_arcs, route_sequence = hmm.viterbi_decode(td, cand_matrix, probs)
    viterbi_time_ms = (time.time() - t_viterbi_start) * 1000
    
    # 5. Route Continuity
    clf_continuity = float(calculate_route_continuity(clf_pred_arcs, arcs))
    hmm_continuity = float(calculate_route_continuity(viterbi_pred_arcs, arcs))
    
    # 6. Accuracy Metrics (evaluated strictly server-side)
    metrics = {
        "classifier_continuity": round(clf_continuity * 100, 2),
        "hmm_continuity": round(hmm_continuity * 100, 2),
        "continuity_improvement": round((hmm_continuity - clf_continuity) * 100, 2),
        "viterbi_latency_ms": round(viterbi_time_ms, 2),
        "total_processing_time_ms": round((time.time() - t_start) * 100, 2),
    }
    
    if has_ground_truth:
        clf_acc = float(np.mean(clf_pred_arcs == true_arcs))
        hmm_acc = float(np.mean(viterbi_pred_arcs == true_arcs))
        miss_rate = float(np.mean([true_arcs[i] not in cand_matrix[i] for i in range(N)]))
        metrics["classifier_top1_accuracy"] = round(clf_acc * 100, 2)
        metrics["hmm_viterbi_accuracy"] = round(hmm_acc * 100, 2)
        metrics["candidate_miss_rate"] = round(miss_rate * 100, 2)
        metrics["ground_truth_route_length"] = len(td["route"])
        
    return {
        "status": "success",
        "trip_id": trip_id,
        "num_observations": N,
        "predicted_route_length": len(route_sequence),
        "predicted_route": route_sequence[:150], # Sample of traversed arc IDs
        "metrics": metrics
    }

@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def render_trip_map(trip_id: str):
    """
    Renders an interactive Folium map comparing:
    - GPS Raw Observations (Blue markers)
    - Ground Truth Reference Path (Green line)
    - Predicted Route Sequence (Purple dashed line)
    """
    trip_path = os.path.join(EVALUATOR_DIR, trip_id)
    if not os.path.exists(trip_path):
        trip_path = os.path.join(TRAIN_DIR, trip_id)
        if not os.path.exists(trip_path):
            raise HTTPException(status_code=404, detail="Trip not found.")
            
    trip_dir = os.path.dirname(trip_path)
    td = load_trip_data(trip_dir, trip_id, load_route=True)
    track = td["track"]
    arcs = td["arcs"]
    nodes = td["nodes"]
    N = len(track)
    
    cand_data = extract_candidates_and_features(td, k_candidates=8)
    probs = model.predict_proba(cand_data["features"])[:, 1].reshape(N, 8)
    viterbi_pred_arcs, route_seq = hmm.viterbi_decode(td, cand_data["cand_matrix"], probs)
    
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))
    
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")
    
    # Subsampled GPS
    step = max(1, len(track) // 250)
    for i in range(0, len(track), step):
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            popup=f"Obs #{i}"
        ).add_to(fmap)
        
    # Ground Truth (if exists)
    if td["route"] is not None:
        gt_lines = []
        for arc_id in td["route"]:
            u, v = arcs[arc_id]
            gt_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        folium.PolyLine(gt_lines, color="#10b981", weight=5, opacity=0.8, tooltip="Ground Truth Reference Route").add_to(fmap)
        
    # Reconstructed Path
    pred_lines = []
    for arc_id in route_seq:
        u, v = arcs[arc_id]
        pred_lines.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
    folium.PolyLine(pred_lines, color="#7c3aed", weight=3, dash_array="6", opacity=0.9, tooltip="PathSnap Reconstructed Path").add_to(fmap)
    
    folium.LayerControl().add_to(fmap)
    return HTMLResponse(content=fmap.get_root().render())

if __name__ == "__main__":
    import uvicorn
    # Default port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

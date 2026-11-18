"""
Hidden Markov Model (HMM) and Viterbi Decoding for Map-Matching and Route Reconstruction.
Preserves candidate-to-road-segment correspondence, incorporates road network topology,
and enforces continuous transitions along graph edges.
"""
import math
import numpy as np
from collections import defaultdict

class MapMatchingHMM:
    """
    HMM for Map-Matching:
    - States: Candidate road segments (arcs) at time step t
    - Emission: Derived from classifier candidate match probability and perpendicular distance
    - Transition: Road connectivity between consecutive candidates (direct connectivity, distance difference, gap handling)
    """
    def __init__(self, sigma_z=4.0, beta=10.0, connect_boost=5.0):
        self.sigma_z = sigma_z
        self.beta = beta
        self.connect_boost = connect_boost

    def build_adjacency(self, arcs):
        """
        Builds head-to-tail adjacency: arc_a -> arc_b if arc_a.to_node == arc_b.from_node
        """
        to_node_map = defaultdict(list)
        for arc_id, (u, v) in enumerate(arcs):
            to_node_map[u].append(arc_id)
            
        # adjacency: arc_id -> list of outgoing arcs
        adj = defaultdict(set)
        for arc_id, (u, v) in enumerate(arcs):
            # Arcs starting where arc_id ends
            for next_arc in to_node_map.get(v, []):
                adj[arc_id].add(next_arc)
        return adj

    def viterbi_decode(self, trip_data, cand_matrix, cand_scores, road_index=None):
        """
        Executes Viterbi decoding over sequence of candidate sets.
        Args:
            trip_data: dictionary with 'track', 'arcs', 'nodes'
            cand_matrix: (N, K) candidate arc indices
            cand_scores: (N, K) predicted probabilities / confidence scores from classifier
            road_index: precomputed spatial index
        Returns:
            predicted_arcs: (N,) optimal sequence of arc IDs
            route_sequence: list of unique connected arc IDs traversed
        """
        track = trip_data["track"]
        arcs = trip_data["arcs"]
        nodes = trip_data["nodes"]
        N, K = cand_matrix.shape
        
        # Build adjacency
        adj = self.build_adjacency(arcs)
        
        # Precompute GPS displacements
        m_lon = road_index["m_lon"] if road_index else 71920.0
        m_lat = road_index["m_lat"] if road_index else 111320.0
        px = track[:, 0] * m_lon
        py = track[:, 1] * m_lat
        pt = track[:, 2]
        
        # Log-Viterbi tables
        # V[t, k]: max log probability to state k at time t
        V = np.full((N, K), -1e9, dtype=np.float64)
        backpointer = np.zeros((N, K), dtype=int)
        
        # Initial step t = 0
        for k in range(K):
            score = max(cand_scores[0, k], 1e-6)
            V[0, k] = math.log(score)
            
        # Recurrence
        for t in range(1, N):
            dt = max(pt[t] - pt[t-1], 0.1)
            dist_gps = math.hypot(px[t] - px[t-1], py[t] - py[t-1])
            is_gap = dt > 5.0 or dist_gps > 100.0
            
            for k_curr in range(K):
                curr_arc = cand_matrix[t, k_curr]
                score_curr = max(cand_scores[t, k_curr], 1e-6)
                emission_log = math.log(score_curr)
                
                best_val = -1e9
                best_prev = 0
                
                for k_prev in range(K):
                    prev_arc = cand_matrix[t-1, k_prev]
                    prev_val = V[t-1, k_prev]
                    
                    if prev_val <= -1e8:
                        continue
                        
                    # Transition model
                    if is_gap:
                        # Large gap: unconstrained transition with slight distance penalty
                        trans_log = -0.5
                    elif curr_arc == prev_arc:
                        # Vehicle stays on the same arc
                        trans_log = 0.5
                    elif curr_arc in adj[prev_arc]:
                        # Direct connected topological neighbor
                        trans_log = 2.0
                    else:
                        # Disconnected / leap penalty
                        trans_log = -5.0
                        
                    total_log = prev_val + trans_log + emission_log
                    if total_log > best_val:
                        best_val = total_log
                        best_prev = k_prev
                        
                V[t, k_curr] = best_val
                backpointer[t, k_curr] = best_prev
                
        # Backtracking
        best_last_k = int(np.argmax(V[N-1, :]))
        pred_k_seq = [best_last_k]
        
        for t in range(N-1, 0, -1):
            best_last_k = backpointer[t, best_last_k]
            pred_k_seq.append(best_last_k)
            
        pred_k_seq.reverse()
        
        predicted_arcs = np.array([cand_matrix[t, pred_k_seq[t]] for t in range(N)], dtype=int)
        
        # Reconstruct route sequence (deduplicate consecutive)
        route_sequence = []
        for arc in predicted_arcs:
            if not route_sequence or route_sequence[-1] != arc:
                route_sequence.append(arc)
                
        return predicted_arcs, route_sequence

def calculate_route_continuity(pred_arcs, arcs):
    """
    Calculates percentage of consecutive predictions that are topologically connected
    (either same arc or arc_t to arc_{t+1} connects head-to-tail in graph).
    """
    if len(pred_arcs) <= 1:
        return 1.0
        
    adj = defaultdict(set)
    for arc_id, (u, v) in enumerate(arcs):
        adj[u].add((arc_id, v))
        
    to_nodes = {arc_id: v for arc_id, (u, v) in enumerate(arcs)}
    from_nodes = {arc_id: u for arc_id, (u, v) in enumerate(arcs)}
    
    continuous_steps = 0
    total_transitions = len(pred_arcs) - 1
    
    for t in range(total_transitions):
        a1 = pred_arcs[t]
        a2 = pred_arcs[t+1]
        if a1 == a2:
            continuous_steps += 1
        elif to_nodes.get(a1) == from_nodes.get(a2):
            continuous_steps += 1
            
    return continuous_steps / total_transitions

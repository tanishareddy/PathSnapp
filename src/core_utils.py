"""
PathSnap Core Utilities: Data loading, spatial projection, geometry, and rich multi-feature candidate extraction.
Features per candidate (12 rich geometric and kinematic features):
  0: perp_distance (m)
  1: distance_to_midpoint (m)
  2: heading_difference (deg, [0, 180])
  3: gps_speed (m/s)
  4: segment_length (m)
  5: projection_ratio (t in [0, 1])
  6: dt (seconds since previous GPS point)
  7: rank_by_dist (0 to k-1)
  8: distance_to_node_u (m)
  9: distance_to_node_v (m)
 10: heading_consistency (cos of angular difference [-1, 1])
 11: projection_endpoint_penalty (|t - 0.5| * 2 in [0, 1])
"""
import os
import math
import numpy as np
from scipy.spatial import cKDTree

M_PER_DEG_LAT = 111320.0

def get_meter_projection(lat_center=49.85):
    cos_lat = math.cos(math.radians(lat_center))
    m_per_deg_lon = M_PER_DEG_LAT * cos_lat
    return m_per_deg_lon, M_PER_DEG_LAT

def load_trip_data(trip_dir, trip_id, load_route=True):
    p = os.path.join(trip_dir, trip_id, trip_id)
    nodes = np.loadtxt(f"{p}.nodes")
    arcs = np.loadtxt(f"{p}.arcs", dtype=int)
    track = np.loadtxt(f"{p}.track")
    
    route = None
    route_file = f"{p}.route"
    if load_route and os.path.exists(route_file):
        route = np.loadtxt(route_file, dtype=int)
        if route.ndim == 0:
            route = np.array([route.item()])
            
    return {
        "trip_id": trip_id,
        "nodes": nodes,
        "arcs": arcs,
        "track": track,
        "route": route
    }

def get_observation_ground_truth(trip_data):
    track = trip_data["track"]
    route = trip_data["route"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]
    
    if route is None or len(route) == 0:
        return np.full(len(track), -1, dtype=int)
        
    lat_center = np.mean(track[:, 1])
    m_lon, m_lat = get_meter_projection(lat_center)
    
    route_u = arcs[route, 0]
    route_v = arcs[route, 1]
    ax = nodes[route_u, 0] * m_lon
    ay = nodes[route_u, 1] * m_lat
    bx = nodes[route_v, 0] * m_lon
    by = nodes[route_v, 1] * m_lat
    dx = bx - ax
    dy = by - ay
    l2 = np.maximum(dx*dx + dy*dy, 1e-6)
    
    px = track[:, 0] * m_lon
    py = track[:, 1] * m_lat
    
    curr_r = 0
    true_arcs = np.empty(len(track), dtype=int)
    
    for i in range(len(track)):
        p_x, p_y = px[i], py[i]
        w_start = max(0, curr_r - 1)
        w_end = min(len(route), curr_r + 30)
        
        t_param = np.clip(((p_x - ax[w_start:w_end])*dx[w_start:w_end] + 
                           (p_y - ay[w_start:w_end])*dy[w_start:w_end]) / l2[w_start:w_end], 0.0, 1.0)
        proj_x = ax[w_start:w_end] + t_param * dx[w_start:w_end]
        proj_y = ay[w_start:w_end] + t_param * dy[w_start:w_end]
        dists = np.hypot(p_x - proj_x, p_y - proj_y)
        
        best_w = np.argmin(dists)
        curr_r = w_start + best_w
        true_arcs[i] = route[curr_r]
        
    return true_arcs

def build_road_spatial_index(nodes, arcs, lat_center=49.85):
    m_lon, m_lat = get_meter_projection(lat_center)
    u_idx = arcs[:, 0]
    v_idx = arcs[:, 1]
    
    ax = nodes[u_idx, 0] * m_lon
    ay = nodes[u_idx, 1] * m_lat
    bx = nodes[v_idx, 0] * m_lon
    by = nodes[v_idx, 1] * m_lat
    
    mid_x = (ax + bx) * 0.5
    mid_y = (ay + by) * 0.5
    
    dx = bx - ax
    dy = by - ay
    l2 = np.maximum(dx*dx + dy*dy, 1e-6)
    length = np.sqrt(l2)
    heading = (np.degrees(np.arctan2(dy, dx))) % 360.0
    
    tree = cKDTree(np.column_stack([mid_x, mid_y]))
    
    return {
        "tree": tree,
        "ax": ax, "ay": ay,
        "bx": bx, "by": by,
        "dx": dx, "dy": dy,
        "l2": l2,
        "length": length,
        "heading": heading,
        "m_lon": m_lon,
        "m_lat": m_lat,
        "num_arcs": len(arcs)
    }

def extract_candidates_and_features(trip_data, road_index=None, k_candidates=20, true_arcs=None):
    track = trip_data["track"]
    N = len(track)
    if road_index is None:
        road_index = build_road_spatial_index(trip_data["nodes"], trip_data["arcs"], np.mean(track[:, 1]))
        
    m_lon = road_index["m_lon"]
    m_lat = road_index["m_lat"]
    px = track[:, 0] * m_lon
    py = track[:, 1] * m_lat
    pt = track[:, 2]
    
    # Query spatial tree
    nn_dists, cand_matrix = road_index["tree"].query(np.column_stack([px, py]), k=k_candidates)
    
    # Calculate GPS speed and heading
    gps_speed = np.zeros(N, dtype=np.float32)
    gps_heading = np.zeros(N, dtype=np.float32)
    dt = np.ones(N, dtype=np.float32)
    
    for i in range(1, N):
        disp_x = px[i] - px[i-1]
        disp_y = py[i] - py[i-1]
        time_diff = max(pt[i] - pt[i-1], 0.1)
        dt[i] = time_diff
        step_dist = math.hypot(disp_x, disp_y)
        gps_speed[i] = step_dist / time_diff
        gps_heading[i] = (math.degrees(math.atan2(disp_y, disp_x))) % 360.0
    gps_speed[0] = gps_speed[1] if N > 1 else 0.0
    gps_heading[0] = gps_heading[1] if N > 1 else 0.0
    
    feature_list = []
    labels = [] if true_arcs is not None else None
    
    ax = road_index["ax"]
    ay = road_index["ay"]
    bx = road_index["bx"]
    by = road_index["by"]
    dx = road_index["dx"]
    dy = road_index["dy"]
    l2 = road_index["l2"]
    arc_len = road_index["length"]
    arc_hd = road_index["heading"]
    
    for i in range(N):
        cands_i = cand_matrix[i]
        c_ax = ax[cands_i]
        c_ay = ay[cands_i]
        c_bx = bx[cands_i]
        c_by = by[cands_i]
        c_dx = dx[cands_i]
        c_dy = dy[cands_i]
        c_l2 = l2[cands_i]
        
        t_param = np.clip(((px[i] - c_ax)*c_dx + (py[i] - c_ay)*c_dy) / c_l2, 0.0, 1.0)
        proj_x = c_ax + t_param * c_dx
        proj_y = c_ay + t_param * c_dy
        perp_dist = np.hypot(px[i] - proj_x, py[i] - proj_y)
        mid_dist = nn_dists[i]
        
        # Dist to endpoints
        dist_u = np.hypot(px[i] - c_ax, py[i] - c_ay)
        dist_v = np.hypot(px[i] - c_bx, py[i] - c_by)
        
        # Angular diff & cosine consistency
        h_diff = np.abs(gps_heading[i] - arc_hd[cands_i])
        h_diff = np.minimum(h_diff, 360.0 - h_diff)
        cos_align = np.cos(np.radians(h_diff))
        endpoint_penalty = np.abs(t_param - 0.5) * 2.0
        
        # Rank by perp_dist
        rank_order = np.argsort(perp_dist)
        ranks = np.empty(k_candidates, dtype=int)
        ranks[rank_order] = np.arange(k_candidates)
        
        for k_idx in range(k_candidates):
            c_id = cands_i[k_idx]
            feat = [
                perp_dist[k_idx],
                mid_dist[k_idx],
                h_diff[k_idx],
                gps_speed[i],
                arc_len[c_id],
                t_param[k_idx],
                dt[i],
                ranks[k_idx],
                dist_u[k_idx],
                dist_v[k_idx],
                cos_align[k_idx],
                endpoint_penalty[k_idx]
            ]
            feature_list.append(feat)
            
            if true_arcs is not None:
                is_match = 1 if (c_id == true_arcs[i]) else 0
                labels.append(is_match)
                
    features = np.array(feature_list, dtype=np.float32)
    if labels is not None:
        labels = np.array(labels, dtype=int)
        
    return {
        "features": features,
        "labels": labels,
        "cand_matrix": cand_matrix,
        "num_obs": N,
        "k_candidates": k_candidates
    }

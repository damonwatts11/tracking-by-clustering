from dataclasses import dataclass
import numpy as np
from tbc.geometry import torus_distance


# =============================================================
# Task 1 — The Instance Data Structure
# =============================================================

@dataclass
class TrackingInstance:
    """
    A container that holds the full graph (nodes + edges + costs)
    ready for the greedy solver in Step 3.

    Fields:
        points  : (M, 2)  — x,y position of every observed point
        times   : (M,)    — which frame each point belongs to
        edges   : (E, 2)  — each row is (i, j), the two node ids connected
        costs   : (E,)    — the score for each edge (positive = join, negative = cut)
        kind    : (E,)    — 0 if within-frame edge, 1 if between-frame edge
        n_nodes : int     — total number of points M
        T       : int     — total number of frames
    """
    points:  np.ndarray   # (M, 2)
    times:   np.ndarray   # (M,)
    edges:   np.ndarray   # (E, 2)
    costs:   np.ndarray   # (E,)
    kind:    np.ndarray   # (E,)  0 = within-frame, 1 = between-frame
    n_nodes: int
    T:       int


# =============================================================
# Task 2 — Nodes Grouped by Frame
# =============================================================

def nodes_by_frame(times: np.ndarray, T: int) -> list:
    """
    For each frame t, return the list of node ids (row indices)
    whose timestamp equals t.

    Args:
        times : (M,) array — frame index for every point
        T     : total number of frames

    Returns:
        A list of length T.
        Entry t is an int array of node ids that belong to frame t.

    Example:
        If times = [0, 0, 1, 0, 1] and T = 2:
        → [[0, 1, 3], [2, 4]]
        Frame 0 has nodes 0, 1, 3
        Frame 1 has nodes 2, 4
    """
    return [np.flatnonzero(times == t) for t in range(T)]


# =============================================================
# Task 3 — Within-Frame Edges (Spatial Gating)
# =============================================================

def within_frame_edges(points: np.ndarray, frames_by_t: list,
                       rho_in: float, L: float):
    """
    For each frame, find all pairs of points closer than rho_in
    on the torus. These are candidate edges for 'same ball, same frame'.

    Args:
        points      : (M, 2) all observed point positions
        frames_by_t : list of length T — each entry is array of node ids in that frame
        rho_in      : spatial gate radius (how close two points must be to get an edge)
        L           : world size (cube_size from SimConfig)

    Returns:
        edges : (E, 2) int array — each row is a pair (i, j) with i < j
        dists : (E,)  float array — torus distance for each edge
    """

    all_edges = []   # will collect edge pairs from every frame
    all_dists = []   # will collect distances from every frame

    for node_ids in frames_by_t:
        # node_ids = e.g. [0, 1, 2, ... 109] for frame 0
        # these are the GLOBAL node ids (row indices in the points array)

        n = len(node_ids)

        # If frame has 0 or 1 points, no pairs possible — skip
        if n < 2:
            continue

        # Get the actual x,y positions of these points
        # pos shape: (n, 2)
        pos = points[node_ids]

        # ---- Compute pairwise torus distance matrix ----
        # We want distance between every pair (i, j) in this frame.
        # pos[:, None, :] shape: (n, 1, 2)
        # pos[None, :, :] shape: (1, n, 2)
        # Broadcasting gives us all pairs at once — shape (n, n, 2)
        # Then torus_distance along last axis gives (n, n) distance matrix

        # a[i] vs b[j] for all i,j:
        a = pos[:, None, :]   # shape (n, 1, 2)
        b = pos[None, :, :]   # shape (1, n, 2)

        # dist_matrix[i, j] = torus distance between local point i and local point j
        # shape: (n, n)
        dist_matrix = torus_distance(a, b, L)

        # ---- Keep only upper triangle where dist < rho_in ----
        # Upper triangle means i < j — avoids storing (3,5) AND (5,3) for same edge
        # np.triu_indices gives us all (row, col) pairs where row < col
        rows, cols = np.triu_indices(n, k=1)   # k=1 means skip diagonal (i != j)

        # Get distances for these upper triangle pairs
        dists_upper = dist_matrix[rows, cols]

        # Keep only pairs closer than rho_in
        mask = dists_upper < rho_in
        local_i = rows[mask]   # local indices within this frame
        local_j = cols[mask]   # local indices within this frame

        if len(local_i) == 0:
            continue

        # ---- Convert local indices back to GLOBAL node ids ----
        # local_i=0 means the first point in this frame → node_ids[0]
        global_i = node_ids[local_i]
        global_j = node_ids[local_j]

        # Stack into (E_frame, 2) array and store
        frame_edges = np.stack([global_i, global_j], axis=1)
        all_edges.append(frame_edges)
        all_dists.append(dists_upper[mask])

    # Concatenate results from all frames into one big array
    if len(all_edges) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)

    edges = np.concatenate(all_edges, axis=0)
    dists = np.concatenate(all_dists, axis=0)

    return edges, dists


# =============================================================
# Task 4 — Between-Frame Edges (Motion Gating + Cyclic Wrap)
# =============================================================

def between_frame_edges(points: np.ndarray, frames_by_t: list,
                        rho_mot: float, L: float, T: int,
                        cyclic: bool = True):
    """
    For each frame t, connect points in frame t to points in frame t+1
    if their torus distance is less than rho_mot.

    If cyclic=True, also connect frame T-1 back to frame 0
    (because time wraps around like a clock).

    Args:
        points      : (M, 2) all observed point positions
        frames_by_t : list of length T — node ids per frame (from Task 2)
        rho_mot     : motion gate radius (how far a ball can move in one frame)
        L           : world size
        T           : total number of frames
        cyclic      : whether to add the wrap-around edge (T-1 -> 0)

    Returns:
        edges : (E, 2) int array — pairs (i, j) with i < j
        dists : (E,) float array — torus distance for each edge
    """

    all_edges = []
    all_dists = []

    # How many consecutive pairs to process?
    # Normally: (0,1), (1,2), ..., (T-2, T-1)  -> that's T-1 pairs
    # If cyclic, also add (T-1, 0)             -> one more pair
    pairs_to_process = list(range(T - 1))   # [0, 1, ..., T-2]
    if cyclic:
        pairs_to_process.append(T - 1)      # add the wrap pair

    for t in pairs_to_process:
        t_next = (t + 1) % T   # for t = T-1, this becomes 0 (the wrap)

        nodes_t      = frames_by_t[t]       # global node ids in frame t
        nodes_t_next = frames_by_t[t_next]  # global node ids in frame t_next

        n_t      = len(nodes_t)
        n_t_next = len(nodes_t_next)

        # Skip if either frame is empty
        if n_t == 0 or n_t_next == 0:
            continue

        pos_t      = points[nodes_t]        # (n_t, 2)
        pos_t_next = points[nodes_t_next]   # (n_t_next, 2)

        # ---- Compute rectangular pairwise distance matrix ----
        # a[i] = position of point i in frame t
        # b[j] = position of point j in frame t_next
        a = pos_t[:, None, :]        # shape (n_t, 1, 2)
        b = pos_t_next[None, :, :]   # shape (1, n_t_next, 2)

        # dist_matrix[i, j] = distance between point i (frame t)
        #                     and point j (frame t_next)
        dist_matrix = torus_distance(a, b, L)   # shape (n_t, n_t_next)

        # ---- Apply the motion gate ----
        # Find ALL (i, j) pairs where distance < rho_mot
        # np.where returns the row indices and column indices separately
        local_i, local_j = np.where(dist_matrix < rho_mot)

        if len(local_i) == 0:
            continue

        dists_kept = dist_matrix[local_i, local_j]

        # ---- Convert local indices to GLOBAL node ids ----
        global_i = nodes_t[local_i]
        global_j = nodes_t_next[local_j]

        # ---- Normalise so that i < j always (avoid duplicate edges) ----
        # Since global_i comes from frame t and global_j from frame t_next,
        # they could be in any numeric order globally. We sort each pair
        # so the smaller node id is always first.
        lo = np.minimum(global_i, global_j)
        hi = np.maximum(global_i, global_j)

        frame_edges = np.stack([lo, hi], axis=1)
        all_edges.append(frame_edges)
        all_dists.append(dists_kept)

    if len(all_edges) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)

    edges = np.concatenate(all_edges, axis=0)
    dists = np.concatenate(all_dists, axis=0)

    return edges, dists


# =============================================================
# Task 5 — Edge Costs (Log-Odds Formula)
# =============================================================

def edge_costs(d: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    """
    Convert distances into edge costs using the log-odds formula.

    Args:
        d     : (E,) array of torus distances for a set of edges
        rho   : the gate radius used for these edges (rho_in or rho_mot)
        alpha : slope — controls how strongly the solver commits
                (start with alpha = 1)

    Returns:
        costs : (E,) array of signed costs
                positive  -> reward joining (d < rho, points are close)
                negative  -> penalize joining (d > rho, points are far)
                zero      -> exactly at the boundary (d == rho)

    Example:
        rho = 0.4, alpha = 1
        d = 0.1  -> cost = 1 * (0.4 - 0.1) = +0.3   (close, reward)
        d = 0.4  -> cost = 1 * (0.4 - 0.4) =  0.0   (boundary)
        d = 0.6  -> cost = 1 * (0.4 - 0.6) = -0.2   (far, penalty)
    """
    return alpha * (rho - d)


# =============================================================
# Task 6 — Build Instance (Assemble + Sanity Check)
# =============================================================

def build_instance(ds, rho_in: float, rho_mot: float,
                   alpha_in: float = 1.0, alpha_mot: float = 1.0,
                   cyclic: bool = True, verbose: bool = True) -> TrackingInstance:
    """
    Build the full TrackingInstance graph from a SyntheticDataset.

    This ties together Tasks 2-5:
      1. Group nodes by frame
      2. Build within-frame edges + costs
      3. Build between-frame edges + costs
      4. Combine everything into one TrackingInstance
      5. Print a sanity-check summary

    Args:
        ds        : SyntheticDataset (from Step 1)
        rho_in    : spatial gate radius for within-frame edges
        rho_mot   : motion gate radius for between-frame edges
        alpha_in  : slope for within-frame edge costs (default 1.0)
        alpha_mot : slope for between-frame edge costs (default 1.0)
        cyclic    : whether to add the wrap-around frame pair (T-1, 0)
        verbose   : if True, print sanity-check stats

    Returns:
        TrackingInstance ready for the Step 3 solver
    """

    T = ds.config.n_timesteps
    L = ds.config.cube_size

    # ---- Step 1: group nodes by frame (Task 2) ----
    frames = nodes_by_frame(ds.times, T)

    # ---- Step 2: within-frame edges (Task 3) ----
    within_edges, within_dists = within_frame_edges(ds.points, frames, rho_in, L)
    within_costs = edge_costs(within_dists, rho_in, alpha_in)
    within_kind = np.zeros(len(within_edges), dtype=int)   # 0 = within-frame

    # ---- Step 3: between-frame edges (Task 4) ----
    between_edges, between_dists = between_frame_edges(
        ds.points, frames, rho_mot, L, T, cyclic=cyclic
    )
    between_costs = edge_costs(between_dists, rho_mot, alpha_mot)
    between_kind = np.ones(len(between_edges), dtype=int)  # 1 = between-frame

    # ---- Step 4: combine everything ----
    edges = np.concatenate([within_edges, between_edges], axis=0)
    costs = np.concatenate([within_costs, between_costs], axis=0)
    kind  = np.concatenate([within_kind, between_kind], axis=0)

    instance = TrackingInstance(
        points=ds.points,
        times=ds.times,
        edges=edges,
        costs=costs,
        kind=kind,
        n_nodes=len(ds.points),
        T=T,
    )

    # ---- Step 5: sanity check / summary ----
    if verbose:
        n_within = len(within_edges)
        n_between = len(between_edges)
        n_total = len(edges)

        frac_pos_within = (within_costs > 0).mean() if n_within > 0 else 0.0
        frac_pos_between = (between_costs > 0).mean() if n_between > 0 else 0.0

        print("===== build_instance() summary =====")
        print(f"|V| (total nodes)           : {instance.n_nodes}")
        print(f"|E_t| (within-frame edges)  : {n_within}")
        print(f"|E_t,t+1| (between edges)   : {n_between}")
        print(f"|E| (total edges)           : {n_total}")
        print(f"Fraction positive (within)  : {frac_pos_within:.2%}")
        print(f"Fraction positive (between) : {frac_pos_between:.2%}")
        print(f"rho_in  = {rho_in:.4f}   alpha_in  = {alpha_in}")
        print(f"rho_mot = {rho_mot:.4f}   alpha_mot = {alpha_mot}")
        print("=====================================")

    return instance

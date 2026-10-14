from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
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

def mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation:  MAD(x) = med(|x_i - med(x)|).

    A robust analogue of the standard deviation: the std sums over all
    the data, so the very large distances introduced by background
    noise inflate it; the median only looks at the central value of the
    ordering, so noise (as long as it stays a minority of the sample)
    does not move it. This is why the fixed "k*std" gate collapsed as
    noise increased: the threshold grew together with the noise.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_threshold(sample: np.ndarray, c: float) -> float:
    """rho = med + c*MAD.  c is the only hyperparameter (dimensionless)."""
    return float(np.median(sample) + c * mad(sample))


# ----------------------------------------------------------------------
# 2. Nearest-neighbour distance samples (torus metric)
# ----------------------------------------------------------------------
# cKDTree(pos, boxsize=L) builds the tree with periodic boundary
# conditions: query() returns minimum-image distances directly,
# consistent with tbc.geometry.torus_distance, without materialising the
# n x n matrix.  Requires points in [0, L), guaranteed by wrap().

def nn_distances_intra(points: np.ndarray, times: np.ndarray,
                       T: int, L: float) -> np.ndarray:
    """
    d_intra(p) = distance (on the torus) from each point to its nearest
    neighbour WITHIN its own frame. Sample aggregated over the T frames.
    This is the quantity whose distribution mixes the internal scale of
    the objects (low values) with the scale of the noise (high values).
    """
    out = []
    for node_ids in nodes_by_frame(times, T):
        if len(node_ids) < 2:
            continue
        pos = points[node_ids]
        tree = cKDTree(pos, boxsize=L)
        # k=2: the nearest neighbour of a point is the point itself (dist 0)
        dists, _ = tree.query(pos, k=2)
        out.append(dists[:, 1])
    return np.concatenate(out) if out else np.array([])


def nn_distances_inter(points: np.ndarray, times: np.ndarray,
                       T: int, L: float) -> np.ndarray:
    """
    d_inter(p) = distance (on the torus) from each point in frame t to
    its nearest neighbour in frame t+1.  Estimates the displacement per
    time step.

    NOTE: only PHYSICALLY consecutive pairs (t, t+1) with t <= T-2.
    The cyclic pair (T-1, 0) used by the tracker does NOT enter the
    sample: the simulation runs forward without temporal periodicity,
    so that pair does not represent a one-step displacement, and
    including it would contaminate the estimator with arbitrarily large
    distances.
    """
    frames = nodes_by_frame(times, T)
    out = []
    for t in range(T - 1):
        ids_t, ids_next = frames[t], frames[t + 1]
        if len(ids_t) == 0 or len(ids_next) == 0:
            continue
        tree = cKDTree(points[ids_next], boxsize=L)
        dists, _ = tree.query(points[ids_t], k=1)
        out.append(dists)
    return np.concatenate(out) if out else np.array([])


# ----------------------------------------------------------------------
# 3. Gate estimation
# ----------------------------------------------------------------------

def estimate_gates(points: np.ndarray, times: np.ndarray, T: int, L: float,
                   c_spatial: float = 3.0, c_motion: float = 3.0):
    """
    rho_s = med(D_s) + c_spatial * MAD(D_s)
    rho_m = med(D_m) + c_motion  * MAD(D_m)

    Returns (rho_s, rho_m). Scale-equivariant: if the point cloud is
    rescaled by lambda, both gates rescale by lambda on their own.
    """
    d_s = nn_distances_intra(points, times, T, L)
    d_m = nn_distances_inter(points, times, T, L)
    return robust_threshold(d_s, c_spatial), robust_threshold(d_m, c_motion)


def estimate_gates_from_dataset(ds, c_spatial: float = 3.0,
                                c_motion: float = 3.0):
    """Convenience: reads points/times/T/L directly from the SyntheticDataset."""
    return estimate_gates(ds.points, ds.times,
                          ds.config.n_timesteps, ds.config.cube_size,
                          c_spatial=c_spatial, c_motion=c_motion)


# ----------------------------------------------------------------------
# 4. Noise pre-filter (optional, recommended with heavy background)
# ----------------------------------------------------------------------

def core_point_mask(points: np.ndarray, times: np.ndarray, T: int, L: float,
                    rho_s: float, min_neighbors: int = 1) -> np.ndarray:
    """
    Boolean mask (M,): True if the point has >= min_neighbors
    neighbours at distance <= rho_s within its own frame.

    Rationale: a point belonging to an object sits in a dense cluster
    (in this simulation, ~n_inliers_per_sphere companions per frame), so
    it has neighbours inside the spatial gate. A point with no
    neighbours can only contribute spurious edges or end up a singleton:
    excluding it from the graph removes the "noise bridges" between
    objects BEFORE the greedy solver can make the mistake (the same
    core/noise logic as DBSCAN).

    Filtered points are NOT removed from the dataset: they are only
    excluded from edge construction and remain singletons, so that the
    labels array stays aligned with ds.labels for the VI.
    """
    mask = np.zeros(len(points), dtype=bool)
    for node_ids in nodes_by_frame(times, T):
        if len(node_ids) < 2:
            continue
        pos = points[node_ids]
        tree = cKDTree(pos, boxsize=L)
        # count neighbours within rho_s (excluding the point itself)
        counts = tree.query_ball_point(pos, r=rho_s, return_length=True) - 1
        mask[node_ids] = counts >= min_neighbors
    return mask
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

def build_instance(ds,
                            c_gate_spatial: float = 3.0,
                            c_gate_motion: float = 3.0,
                            c_cost_spatial: float = 2.0,
                            c_cost_motion: float = 2.0,
                            alpha_in: float = 1.0,
                            alpha_mot: float = 1.0,
                            min_neighbors: int | None = 3,
                            cyclic: bool = True,
                            verbose: bool = True) -> TrackingInstance:
    """
    Assembles the full TrackingInstance. Unlike a fixed-radius
    construction:

      1) rho_gate and rho_cost are ESTIMATED from the data
         (med + c*MAD) instead of being set by hand;
      2) rho_gate (c_gate_*) and rho_cost (c_cost_*) are SEPARATE:
           - the gate admits edges up to med + c_gate*MAD  (recall)
           - the cost crosses zero at   med + c_cost*MAD   (precision)
         with c_cost < c_gate, so that negative-cost edges exist and
         GAEC solves a genuine multicut instead of returning connected
         components;
      3) optionally excludes noise points from the graph (min_neighbors;
         None to disable).

    Requirement: c_cost_* <= c_gate_* (otherwise the gate clips away the
    negative edges and we are back to the original problem).
    """
    assert c_cost_spatial <= c_gate_spatial and c_cost_motion <= c_gate_motion, \
        "c_cost must be <= c_gate: the gate must admit negative-cost edges"

    T = ds.config.n_timesteps
    L = ds.config.cube_size
    points, times = ds.points, ds.times

    # ---- 1) d1 samples and the four radii -----------------------------
    d_s = nn_distances_intra(points, times, T, L)
    d_m = nn_distances_inter(points, times, T, L)

    rho_gate_in  = robust_threshold(d_s, c_gate_spatial)
    rho_gate_mot = robust_threshold(d_m, c_gate_motion)
    rho_cost_in  = robust_threshold(d_s, c_cost_spatial)
    rho_cost_mot = robust_threshold(d_m, c_cost_motion)

    # ---- 2) group nodes by frame; optional noise pre-filter -----------
    frames = nodes_by_frame(times, T)
    if min_neighbors is not None:
        mask = core_point_mask(points, times, T, L,
                               rho_gate_in, min_neighbors=min_neighbors)
        frames = [ids[mask[ids]] for ids in frames]
        n_filtered = int((~mask).sum())
    else:
        n_filtered = 0

    # ---- 3) edges from the generous gate, costs with the inner zero ---
    w_edges, w_dists = within_frame_edges(points, frames, rho_gate_in, L)
    w_costs = edge_costs(w_dists, rho_cost_in, alpha_in)

    b_edges, b_dists = between_frame_edges(points, frames, rho_gate_mot,
                                           L, T, cyclic=cyclic)
    b_costs = edge_costs(b_dists, rho_cost_mot, alpha_mot)

    edges = np.concatenate([w_edges, b_edges], axis=0)
    costs = np.concatenate([w_costs, b_costs], axis=0)
    kind = np.concatenate([np.zeros(len(w_edges), dtype=int),
                           np.ones(len(b_edges), dtype=int)])

    instance = TrackingInstance(points=points, times=times, edges=edges,
                                costs=costs, kind=kind,
                                n_nodes=len(points), T=T)

    if verbose:
        fp_w = (w_costs > 0).mean() if len(w_costs) else 0.0
        fp_b = (b_costs > 0).mean() if len(b_costs) else 0.0
        print("===== build_instance() summary =====")
        print(f"rho_gate_in  = {rho_gate_in:.4f}   rho_cost_in  = {rho_cost_in:.4f}")
        print(f"rho_gate_mot = {rho_gate_mot:.4f}   rho_cost_mot = {rho_cost_mot:.4f}")
        print(f"points excluded by noise filter : {n_filtered}")
        print(f"|E_t| = {len(w_edges)}   |E_t,t+1| = {len(b_edges)}   |E| = {len(edges)}")
        print(f"Positive fraction (within)  : {fp_w:.2%}   <- should NOT be 100%")
        print(f"Positive fraction (between) : {fp_b:.2%}")
        print("=============================================")

    return instance


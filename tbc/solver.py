import numpy as np
import heapq
from collections import defaultdict


# =============================================================
# Helper — Union-Find (Disjoint Set Union)
# =============================================================

class UnionFind:
    """
    Tracks which nodes belong to the same cluster.
    find(i)    → root/representative of i's cluster
    union(i,j) → merges the two clusters
    """

    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.size   = np.ones(n, dtype=int)

    def find(self, i: int) -> int:
        # Path compression
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> int:
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            return ri
        # Union by size: smaller joins larger
        if self.size[ri] < self.size[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        self.size[ri] += self.size[rj]
        return ri


# =============================================================
# Task 7 — Unified GAEC on the full graph (heap-based, fast)
# =============================================================

def gaec(instance) -> np.ndarray:
    """
    Greedy Additive Edge Contraction on the ENTIRE graph at once.

    All within-frame and between-frame edges compete in one pool.
    Uses a max-heap for O(E log E) performance instead of O(E^2).

    Returns:
        labels : (M,) int array — track id per point
    """

    n     = instance.n_nodes
    edges = instance.edges   # (E, 2)
    costs = instance.costs   # (E,)

    uf = UnionFind(n)

    # join_cost[(a,b)] = summed cost of all edges between cluster a and b
    # Always stored as (min_root, max_root) → one entry per pair
    join_cost = defaultdict(float)

    # Build initial join costs from all positive edges
    for idx in range(len(edges)):
        c = costs[idx]
        if c <= 0:
            continue
        i, j = int(edges[idx, 0]), int(edges[idx, 1])
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        key = (min(ri, rj), max(ri, rj))
        join_cost[key] += c

    # Max-heap: Python's heapq is a min-heap, so we push negative costs
    # Each entry: (-cost, a, b)
    # We use lazy deletion: entries become stale after merges, we skip them
    heap = []
    for (a, b), val in join_cost.items():
        heapq.heappush(heap, (-val, a, b))

    # Greedy merge loop
    while heap:
        neg_val, a, b = heapq.heappop(heap)
        val = -neg_val

        if val <= 0:
            break   # heap is sorted; nothing better remains

        # Find current roots (they may have changed)
        ra = uf.find(a)
        rb = uf.find(b)

        if ra == rb:
            continue   # stale entry — already merged

        # Check if the stored cost is still current
        key = (min(ra, rb), max(ra, rb))
        if key not in join_cost:
            continue   # stale
        if abs(join_cost[key] - val) > 1e-9:
            # The cost changed since this heap entry was pushed — re-push
            # the current value and skip this stale entry
            heapq.heappush(heap, (-join_cost[key], key[0], key[1]))
            continue

        # Remove from dict — we're processing this merge now
        del join_cost[key]

        # Perform the merge
        new_root = uf.union(ra, rb)
        old_root = rb if new_root == ra else ra

        # Fold old_root's connections into new_root
        # Collect all keys involving old_root
        affected = [k for k in list(join_cost.keys())
                    if k[0] == old_root or k[1] == old_root]

        for k in affected:
            x, y = k
            neighbor = y if x == old_root else x
            nb_root = uf.find(neighbor)

            old_val = join_cost.pop(k)

            if nb_root == new_root:
                # Neighbor is now inside the merged cluster — discard
                continue

            new_key = (min(new_root, nb_root), max(new_root, nb_root))
            join_cost[new_key] += old_val

            # Push updated entry to heap (lazy: old entries become stale)
            if join_cost[new_key] > 0:
                heapq.heappush(heap, (-join_cost[new_key], new_key[0], new_key[1]))
            else:
                # Negative accumulated cost — remove so it's never selected
                del join_cost[new_key]

    # Extract labels: map root ids → clean 0-indexed track ids
    raw_labels = np.array([uf.find(i) for i in range(n)])
    _, inverse  = np.unique(raw_labels, return_inverse=True)
    return inverse


# =============================================================
# Task 8 — Wrapper: labels + objective value + summary
# =============================================================

def greedy_solve(instance):
    """
    Run unified GAEC and return (labels, objective).

    labels    : (M,) int — predicted track id per point
    objective : float    — sum of costs of joined edges
    """

    labels = gaec(instance)

    # Objective: sum costs where both endpoints are in same cluster
    i_nodes = instance.edges[:, 0]
    j_nodes = instance.edges[:, 1]
    joined  = labels[i_nodes] == labels[j_nodes]
    objective = float(instance.costs[joined].sum())

    # Summary
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters  = len(unique_labels)
    n_large     = int((counts >= 10).sum())
    n_singleton = int((counts == 1).sum())

    print("===== greedy_solve() summary =====")
    print(f"Total nodes           : {instance.n_nodes}")
    print(f"Total clusters found  : {n_clusters}")
    print(f"Large clusters (>=10) : {n_large}   <- expected ~K real tracks")
    print(f"Singleton clusters    : {n_singleton}   <- background/noise points")
    print(f"Objective value       : {objective:.4f}")
    print(f"Note: plain multicut — clustering constraints satisfied,")
    print(f"      no-split/no-join are relaxed (standard GAEC trade-off).")
    print("==================================")

    return labels, objective


# =============================================================
# Task 9 — Variation of Information
# =============================================================

def variation_of_information(labels_true: np.ndarray,
                              labels_pred: np.ndarray) -> float:
    """
    VI = H(true) + H(pred) - 2 * I(true; pred)
    VI = 0  → perfect match
    Smaller is better. Units: nats.
    """

    n = len(labels_true)
    assert len(labels_pred) == n

    # Map to consecutive ids
    _, true_inv = np.unique(labels_true, return_inverse=True)
    _, pred_inv = np.unique(labels_pred, return_inverse=True)

    n_true = true_inv.max() + 1
    n_pred = pred_inv.max() + 1

    # Contingency table
    contingency = np.zeros((n_true, n_pred), dtype=float)
    np.add.at(contingency, (true_inv, pred_inv), 1.0)

    r = contingency / n          # joint
    p = r.sum(axis=1)            # marginal true
    q = r.sum(axis=0)            # marginal pred

    # H(true)
    H_true = -np.sum(p[p > 0] * np.log(p[p > 0]))

    # H(pred)
    H_pred = -np.sum(q[q > 0] * np.log(q[q > 0]))

    # I(true; pred)
    outer = np.outer(p, q)
    mask  = r > 0
    I     = np.sum(r[mask] * np.log(r[mask] / outer[mask]))

    vi = H_true + H_pred - 2.0 * I
    return float(max(vi, 0.0))

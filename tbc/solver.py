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

def gaec(inst) -> np.ndarray:
    """
    Greedy Additive Edge Contraction on the ENTIRE graph at once.

    All within-frame and between-frame edges compete in one pool.
    Uses a max-heap for O(E log E) performance instead of O(E^2).

    Returns:
        labels : (M,) int array — track id per point
    """

    n, edges, costs = inst.n_nodes, inst.edges, inst.costs
    uf = UnionFind(n)
    join = defaultdict(float)
    adj = defaultdict(set)                        # root -> neighbours (roots)
    for idx in range(len(edges)):
        i, j = int(edges[idx, 0]), int(edges[idx, 1])
        key = (min(i, j), max(i, j))
        join[key] += float(costs[idx])
        adj[i].add(j); adj[j].add(i)
    heap = [(-v, a, b) for (a, b), v in join.items() if v > 0]
    heapq.heapify(heap)
    while heap:
        nv, a, b = heapq.heappop(heap)
        v = -nv
        if v <= 0: break
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb: continue
        key = (min(ra, rb), max(ra, rb))
        if key not in join or abs(join[key] - v) > 1e-9:
            if key in join and join[key] > 0:
                heapq.heappush(heap, (-join[key], key[0], key[1]))
            continue
        del join[key]
        new_root = uf.union(ra, rb)
        old_root = rb if new_root == ra else ra
        adj[new_root].discard(old_root); adj[old_root].discard(new_root)
        for nb in adj.pop(old_root, ()):         # fold ONLY neighbours of old_root
            k = (min(old_root, nb), max(old_root, nb))
            if k not in join: continue
            val = join.pop(k)
            adj[nb].discard(old_root)
            nbr = uf.find(nb)
            if nbr == new_root: continue
            nk = (min(new_root, nbr), max(new_root, nbr))
            join[nk] += val
            adj[new_root].add(nbr); adj[nbr].add(new_root)
            if join[nk] > 0:
                heapq.heappush(heap, (-join[nk], nk[0], nk[1]))
    raw = np.array([uf.find(i) for i in range(n)])
    return np.unique(raw, return_inverse=True)[1]


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

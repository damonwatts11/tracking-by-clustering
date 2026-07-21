# Tracking-by-Clustering (TBC)

**Multi-Object Tracking as a single graph-clustering problem — simulated spheres on a 2D torus, tracked with Greedy Additive Edge Contraction (GAEC).**

Team project for the _Machine Learning for Computer Vision_ (MLCV) course at TU Dresden (10 ECTS).
Supervisor: Prof. Bjoern Andres · Consultant: David Stein
Team: Taha ([`taha7672`](https://github.com/taha7672)) & Daniel Watts ([`damonwatts11`](https://github.com/damonwatts11))

---

## 1. What this project does

Classical tracking pipelines detect objects per frame, then link detections over time in a separate step. **Tracking-by-Clustering** does both at once: every observed point (across _all_ frames) becomes a node in one big graph, and a single clustering of that graph simultaneously answers

- _"Which points in the same frame belong to the same object?"_ (detection / grouping), and
- _"Which points in different frames belong to the same object?"_ (association / tracking).

The full pipeline:

1. **Synthesis** — simulate K spheres moving on a 2D torus (periodic boundaries) with elastic collisions, then generate noisy point-cloud observations plus background clutter.
2. **Instance construction** — build a graph over all observed points: within-frame and between-frame edges. Gate radii and edge-cost thresholds are **estimated from the data** (robust `median + c·MAD` statistics), and each edge gets a **signed** cost — positive rewards joining, negative penalises it.
3. **Solver** — cluster the graph with a heap-based GAEC (a greedy heuristic for the minimum-cost multicut / correlation-clustering problem). Each resulting cluster is one recovered track.
4. **Evaluation** — compare predicted clusters against ground-truth identities using **Variation of Information (VI)**, plus ARI/NMI as supporting metrics, across motion models, noise levels, and seeds.

At the default configuration (4 spheres, 60 frames, 20 inliers/sphere, 50 background points/frame) the pipeline handles **~7,800 nodes**. A DBSCAN-style pre-filter drops the isolated clutter (~3,000 points at defaults) before edges are built, leaving roughly **1.1×10⁵ edges** for a single GAEC pass.

---

## 2. Repository structure

```
tracking-by-clustering/
├── tbc/                     # the Python package
│   ├── __init__.py
│   ├── geometry.py          # torus math: wrap, minimum-image displacement, torus distance
│   ├── motion.py            # initial states, motion step, Trajectory, simulate()
│   ├── collision.py         # elastic sphere–sphere collision resolution
│   ├── observation.py       # noisy inlier sampling + uniform background clutter (label -1)
│   ├── synthesis.py         # SimConfig, SyntheticDataset, generate/save/load dataset
│   ├── instance.py          # robust MAD gates, noise pre-filter, signed costs, build_instance()
│   ├── solver.py            # UnionFind, gaec(), greedy_solve(), variation_of_information()
│   ├── viz.py               # basic animation / world visualisation
│   └── viz_report.py        # report-grade figures (noise curves, heatmaps, panels, …)
├── app.py                   # interactive Streamlit explorer
└── README.md
```

---

## 3. Installation

Requirements: Python ≥ 3.10.

```bash
git clone https://github.com/damonwatts11/tracking-by-clustering.git
cd tracking-by-clustering
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy` (torus-aware nearest-neighbour gating), `plotly`, `streamlit`, `jupyter`/`ipykernel`. No compiled extensions and **no external solver** (see §8 on Gurobi).

---

## 4. Quick start

### End-to-end in Python

```python
from tbc.synthesis import SimConfig, generate_dataset
from tbc.instance  import build_instance
from tbc.solver    import greedy_solve, variation_of_information

cfg  = SimConfig(n_spheres=4, n_timesteps=60, obs_noise_std=0.1, seed=0)
ds   = generate_dataset(cfg)

# gates and cost thresholds are estimated from the point cloud itself;
# the only knobs are the dimensionless c_gate / c_cost multipliers.
inst        = build_instance(ds)          # defaults: c_gate=3, c_cost=2, min_neighbors=3
labels_pred = greedy_solve(inst)

# VI is reported on inliers only (ground-truth label >= 0; background is -1)
mask = ds.labels >= 0
vi   = variation_of_information(ds.labels[mask], labels_pred[mask])
print(f"VI = {vi:.4f}")   # 0.0 = perfect recovery
```

### Notebook

### Interactive explorer

```bash
streamlit run app.py
```

The app exposes every `SimConfig` parameter plus the gate controls (`c_gate`, `c_cost`, and the noise pre-filter's `min neighbours`) in the sidebar, and provides synchronized views: animated ground-truth world, radar-vs-truth point clouds, the gated instance graph between consecutive frames, the GAEC tracking result with per-track bounding circles, and evaluation tabs (per-run metrics, edge quality, noise sweep).

---

## 5. Step-by-step design

### Step 1 — Data synthesis (`geometry`, `motion`, `collision`, `observation`, `synthesis`)

- **World:** a square `[0, L)²` with periodic boundaries — topologically a **2D torus**. All distances use the minimum-image convention (`torus_distance`), so an object leaving the right edge re-enters on the left, and distance is measured "the short way around".
- **Motion models:**
  - _Ballistic_ (`motion_noise_std = 0.0`) — constant velocity between collisions.
  - _Random walk_ (`motion_noise_std > 0`, default 0.05) — Gaussian velocity perturbations each step.
- **Collisions:** elastic sphere–sphere collisions (`elasticity = 1.0` by default) resolved on the torus.
- **Observations:** per frame, each sphere emits `n_inliers_per_sphere` points drawn from a Gaussian around its center (`obs_noise_std`) and labeled `k`; plus `n_background` uniformly distributed clutter points labeled `-1`.
- **Reproducibility:** everything is driven by a single `seed` through `np.random.default_rng`; datasets can be saved/loaded as compressed `.npz` with the full config embedded.

All parameters live in one dataclass:

| `SimConfig` field      | default | meaning                              |
| ---------------------- | ------- | ------------------------------------ |
| `cube_size` (L)        | 10.0    | torus side length                    |
| `n_spheres` (K)        | 4       | number of objects                    |
| `radius`               | 0.5     | sphere radius (collision geometry)   |
| `elasticity`           | 1.0     | collision restitution                |
| `speed`                | 1.0     | initial speed magnitude              |
| `motion_noise_std`     | 0.05    | 0 = ballistic, >0 = random walk      |
| `n_timesteps` (T)      | 60      | number of frames                     |
| `dt`                   | 0.1     | time step                            |
| `n_inliers_per_sphere` | 20      | observed points per sphere per frame |
| `n_background`         | 50      | clutter points per frame             |
| `obs_noise_std`        | 0.1     | Gaussian observation noise           |
| `seed`                 | 0       | RNG seed                             |

### Step 2 — Instance construction (`instance.py`)

This step was reworked to make the gates **self-calibrating** and the clustering a **genuine multicut**. Three ideas do the work.

**(a) Robust, data-driven gates — `median + c·MAD`.** Instead of hard-coding radii from the known noise level, the gates are estimated from the point cloud itself. Two distance samples are collected with `scipy.spatial.cKDTree(pos, boxsize=L)` (periodic / torus nearest-neighbour queries that agree with `torus_distance`, without ever forming the full n×n matrix):

- `d_s` — each point's nearest-neighbour distance _within its own frame_ (mixes the tight internal object scale with the loose noise scale);
- `d_m` — each point's nearest-neighbour distance _to the next frame_ (an estimate of per-step displacement).

A radius is then `ρ = median(sample) + c · MAD(sample)` (MAD = median absolute deviation). Why median/MAD and not mean/std: the background injects a minority of very large distances that **inflate the standard deviation**, so an old-style `k·std` gate _grows with the noise_ — which is exactly why the previous fixed-σ gate collapsed at high noise. The median and MAD ignore that outlier minority, so the gate stays anchored to the true object scale. The gates are also **scale-equivariant**: rescale the whole cloud by λ and both radii rescale by λ on their own — nothing absolute to re-tune.

**(b) Two radii per axis — a gate radius and a cost radius.** This is what turns the graph into a real minimum-cost multicut:

- `ρ_gate = med + c_gate·MAD` (default `c_gate = 3`) decides which pairs _become edges at all_ — kept generous so no true edge is dropped (recall).
- `ρ_cost = med + c_cost·MAD` (default `c_cost = 2`) is the **zero-crossing** of the cost. Each edge cost is `c(i, j) = α · (ρ_cost − d)`.

Because `c_cost < c_gate`, every pair with `ρ_cost < d < ρ_gate` is _inside_ the gate but _beyond_ the cost zero-crossing, so it carries a **negative** cost. The instance therefore contains genuine repulsive edges (≈ 22–27 % of all edges at defaults), and GAEC solves an actual multicut rather than degenerating to connected components. An assertion enforces `c_cost ≤ c_gate` (otherwise the gate would clip the negative edges and the old degenerate behaviour would return). Separate multipliers exist for the spatial and motion axes (`c_gate_spatial/motion`, `c_cost_spatial/motion`).

**(c) DBSCAN-style noise pre-filter.** Before any edge is built, `core_point_mask` keeps a point only if it has at least `min_neighbors` neighbours within `ρ_gate_in` in its own frame (default `min_neighbors = 3`). An isolated clutter point can only ever form spurious "bridge" edges between real objects, so it is excluded from the graph _before_ the greedy solver can merge through it. At defaults this drops ≈ 3,000 of the 7,800 points — essentially the entire uniform background — which removes the spacetime-percolation failure path. Filtered points are **not deleted**: they remain in the dataset as singleton clusters, so the predicted-label array stays aligned with ground truth for VI.

**Resulting graph (default config):** ~7,800 nodes; ~32.8k within-frame edges and ~79.3k between-frame edges (~1.1×10⁵ total). Between-frame edges include **cyclic wrap edges** between the last and first frame, so the spacetime domain (x, y, t) is a 3-torus. Everything is assembled into a `TrackingInstance` (points, times, edges, signed costs, per-edge kind, node/frame counts).

### Step 3 — Solver (`solver.py`)

- **UnionFind** with path compression and union-by-size for near-constant-time cluster membership.
- **`gaec(instance)`** — Greedy Additive Edge Contraction over the **entire graph at once** (within-frame and between-frame edges compete in a single pool):
  1. Aggregate edge costs per cluster pair into `join_cost`.
  2. Push all positive pair-costs onto a max-heap (lazy deletion for stale entries).
  3. Repeatedly pop the highest-value pair; if still valid and positive, contract the two clusters and merge their cost tables.
  4. Stop when no positive join remains.

  Because the instance now carries both positive and negative edges (positive fraction ≈ 73–78 % at defaults), this is a real greedy multicut: contracting a pair can turn a previously-positive join negative once repulsive edges are folded in, so the solver stops short of merging everything. Complexity **O(E log E)** — the full ~1.1×10⁵-edge instance solves in a single pass with no decomposition.

- **Background handling:** clutter that survives the pre-filter usually has no strong edges and naturally ends up as singleton clusters — no special-case logic needed.
- **`greedy_solve(instance)`** wraps `gaec` and prints a run summary.
- **`variation_of_information(labels_true, labels_pred)`** computes VI from a contingency table: `VI = H(true) + H(pred) − 2·I(true; pred)`, in nats. VI = 0 means identical partitions; lower is better. VI is reported **inliers-only** (restricted to ground-truth label ≥ 0).

### Step 4 — Evaluation

Experiment harness sweeps **2 motion models × 4 observation-noise levels (0.05, 0.10, 0.20, 0.40) × 3 seeds**, reporting VI (primary, as assigned), with ARI and NMI as secondary metrics (`viz_report.partition_metrics`).

---

## 6. Results

Mean **inliers-only VI** over seeds {0, 1, 2} at the default configuration, with the new data-driven gates:

| σ_obs | mean VI (nats) | behaviour                                                                |
| ----- | -------------- | ------------------------------------------------------------------------ |
| 0.05  | **≈ 0.09**     | tracks recovered; mild over-segmentation into a few low-mass fragments   |
| 0.10  | **≈ 0.18**     | still essentially correct, a little more fragmentation                   |
| 0.20  | **≈ 0.38**     | fragmentation grows smoothly                                             |
| 0.40  | **≈ 1.51**     | heavy fragmentation (hundreds of clusters), VI just past the ln 4 anchor |

**Degradation is now smooth and monotone — the catastrophic collapse is gone.** Two things changed relative to the earlier fixed-gate design:

- The old pipeline snapped to a single merged cluster at high noise, hitting the analytic ceiling `VI = ln 4 ≈ 1.386` (a balanced 4-way ground truth collapsed into one cluster). The robust gates no longer inflate with noise and the pre-filter removes the clutter bridges, so **that all-merge collapse no longer happens**.
- The new residual error is the _opposite_ regime — **over-segmentation**. Even at σ = 0.05 the solver splits each track into a handful of pieces (tens of predicted clusters vs. 4 true), but those extra clusters carry little probability mass, so VI stays near zero. At σ = 0.40 the error is fragmentation into hundreds of clusters, which is why VI slightly _exceeds_ ln 4 (that anchor describes the merge regime, not this split regime).

**Trade-off, stated honestly:** the pipeline no longer reaches exactly VI = 0 at low noise (the previous design did), but it degrades gracefully instead of collapsing, and it never merges distinct objects into one track.

_(Numbers above were produced by re-running the sweep against the current `build_instance`. Regenerate them if you retune `c_gate`, `c_cost`, or `min_neighbors`.)_

---

## 7. Visualization & reporting (`viz_report.py`, `app.py`)

Report-grade figures used in the final talk:

- **Noise-degradation curve** — VI (and ARI/NMI) vs. σ_obs, one line per motion model, seed-based error bands.
- **Contingency heatmap** — true sphere labels vs. predicted clusters (inliers only).
- **Before/after panel** — raw noisy detections vs. points colored by recovered track with start/end markers.
- **Edge-cost histograms** — same-sphere vs. cross-sphere edge cost distributions (now showing the negative tail).
- **Collapse small-multiples** — predictions at low/mid/high noise side by side.
- **Gated-graph frame view** — nodes and gated edges of `G_t`, `G_{t+1}` and the between-frame edge set.
- **Scalability plot** — runtime vs. |E|, with the reference ILP's decomposition ceiling marked.

Deliberately **not** included: interpolated "recovered trajectory" lines (GAEC clusters observed points; it does not interpolate positions) and energy-conservation plots under nonzero motion noise (energy is not conserved by construction).

---

## 8. Design decisions

**GAEC instead of ILP (Gurobi).** With Prof. Andres's approval, the two-person team scope replaces the exact minimum-cost-multicut ILP with the GAEC heuristic. This turned out to be a substantive strength, not just a simplification: the reference Gurobi setup was limited by its trial license to sub-200-variable subproblems and had to decompose the instance, weakening any global-optimality claim. GAEC processes the full ~7,800-node / ~10⁵-edge graph **natively in one pass**.

**Robust, self-calibrating gates.** Radii are estimated as `median + c·MAD` of the observed nearest-neighbour distances rather than derived from the (in practice unknown) noise level. This is scale-equivariant and, crucially, does not inflate with background clutter — fixing the noise-driven gate growth that caused the earlier design to collapse at high σ.

**Two-radius gate/cost split → a genuine multicut.** A generous `ρ_gate` admits edges (recall) while a tighter `ρ_cost` sets the cost zero-crossing (precision). Pairs between the two thresholds carry negative cost, so the instance contains real repulsive edges and GAEC solves an actual multicut instead of returning connected components.

**Noise pre-filter as a percolation guard.** Excluding isolated points (DBSCAN core/noise logic) before edge construction removes the clutter "bridges" that used to let the greedy solver merge distinct objects, without deleting any point from the evaluation.

**Unified single-pass clustering.** One GAEC over all edges simultaneously (within-frame and between-frame in the same pool), rather than an artificial cluster-per-frame-then-link two-stage design.

---

## 9. Reproducibility

- A single `seed` drives all randomness (simulation + observation) through `np.random.default_rng`.
- Gate estimation is deterministic given the point cloud (`median + c·MAD`), so a fixed seed + fixed `c_gate`/`c_cost`/`min_neighbors` reproduces the exact instance.
- Experiment sweeps run over fixed seed sets {0, 1, 2}.
- Datasets round-trip via `save_dataset` / `load_dataset` (`.npz`, config included).

---

## 10. Acknowledgements

- **Prof. Bjoern Andres** — supervision, project definition, scope-reduction approval.
- **David Stein** — consulting throughout the project.
- Course: _Machine Learning for Computer Vision_, TU Dresden. Final talks: July 21, 2026.

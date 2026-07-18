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
2. **Instance construction** — build a graph over all observed points: within-frame edges and between-frame edges, each with a signed real-valued cost.
3. **Solver** — cluster the graph with a heap-based GAEC (a greedy heuristic for the minimum-cost multicut / correlation-clustering problem). Each resulting cluster is one recovered track.
4. **Evaluation** — compare predicted clusters against ground-truth identities using **Variation of Information (VI)**, plus ARI/NMI as supporting metrics, across motion models, noise levels, and seeds.

At the default configuration (4 spheres, 60 frames, 20 inliers/sphere, 50 background points/frame) the pipeline handles **~7,800 nodes** and **~145,000 edges** in a single GAEC pass.

---

## 2. Repository structure

```
tracking-by-clustering/
├── tbc/                     # the Python package
│   ├── __init__.py
│   ├── geometry.py          # torus math: wrap, minimum-image displacement, torus distance
│   ├── motion.py            # initial states, motion step, Trajectory, simulate()
│   ├── collision.py         # elastic sphere–sphere collision resolution
│   ├── observation.py       # noisy inlier sampling + uniform background clutter
│   ├── synthesis.py         # SimConfig, SyntheticDataset, generate/save/load dataset
│   ├── instance.py          # TrackingInstance, gating, edge costs, build_instance()
│   ├── solver.py            # UnionFind, gaec(), greedy_solve(), variation_of_information()
│   ├── viz.py               # basic animation / world visualisation
│   └── viz_report.py        # report-grade figures (noise curves, heatmaps, panels, …)
├── main.ipynb               # end-to-end walkthrough of all four steps
├── app.py                   # interactive Streamlit explorer
├── VIZ_TODO.md              # prioritised visualization plan for the final talk
└── README.md
```

---

## 3. Installation

Requirements: Python ≥ 3.10.

```bash
git clone https://github.com/damonwatts11/tracking-by-clustering.git
cd tracking-by-clustering
pip install numpy matplotlib plotly streamlit jupyter
```

No compiled dependencies and **no external solver** (see §8 on Gurobi).

---

## 4. Quick start

### End-to-end in Python

```python
from tbc.synthesis import SimConfig, generate_dataset
from tbc.instance  import build_instance
from tbc.solver    import greedy_solve, variation_of_information

cfg = SimConfig(n_spheres=4, n_timesteps=60, obs_noise_std=0.1, seed=0)
ds  = generate_dataset(cfg)

# gates: ρ_in from observation noise, ρ_mot from max per-frame displacement
rho_in  = 4 * cfg.obs_noise_std + 0.05
rho_mot = cfg.speed * cfg.dt + 4 * cfg.obs_noise_std + 0.1

inst        = build_instance(ds, rho_in=rho_in, rho_mot=rho_mot)
labels_pred = greedy_solve(inst)

vi = variation_of_information(ds.labels, labels_pred)
print(f"VI = {vi:.4f}")   # 0.0 = perfect recovery
```

### Notebook

`main.ipynb` walks through all four steps with intermediate outputs and figures.

### Interactive explorer

```bash
streamlit run app.py
```

The app exposes every `SimConfig` parameter and both gate radii in the sidebar, and provides five synchronized views: animated ground-truth world, radar-vs-truth point clouds, the gated instance graph between consecutive frames, the GAEC tracking result with per-track bounding circles, and evaluation tabs (per-run metrics, edge quality, noise sweep).

---

## 5. Step-by-step design

### Step 1 — Data synthesis (`geometry`, `motion`, `collision`, `observation`, `synthesis`)

- **World:** a square `[0, L)²` with periodic boundaries — topologically a **2D torus**. All distances use the minimum-image convention (`torus_distance`), so an object leaving the right edge re-enters on the left, and distance is measured "the short way around".
- **Motion models:**
  - _Ballistic_ (`motion_noise_std = 0.0`) — constant velocity between collisions.
  - _Random walk_ (`motion_noise_std > 0`, default 0.1) — Gaussian velocity perturbations each step.
- **Collisions:** elastic sphere–sphere collisions (`elasticity = 1.0` by default) resolved on the torus.
- **Observations:** per frame, each sphere emits `n_inliers_per_sphere` points drawn from a Gaussian around its center (`obs_noise_std`), plus `n_background` uniformly distributed clutter points labeled as background.
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

- **Nodes:** every observed point across all frames (default run: ~7,800).
- **Within-frame edges** (~46,400): all pairs in the same frame with torus distance `d < ρ_in` (spatial gate).
- **Between-frame edges** (~98,700): all pairs in consecutive frames with `d < ρ_mot` (motion gate), including **cyclic wrap edges** between the last and first frame (the spacetime domain (x, y, t) is a 3-torus).
- **Edge costs** use a linear log-odds-style formula:

  ```
  c(i, j) = α · (ρ − d)
  ```

  Points much closer than the gate get strongly positive costs (join), points near the gate boundary get costs near zero. Because gating already removes every pair with `d ≥ ρ`, **all retained edge costs are positive** — the "cut" signal is expressed by _absence_ of an edge rather than a negative cost.

- Everything is assembled into a `TrackingInstance` (points, times, edges, costs, edge kind, node/frame counts). Total: **~145,000 edges** at defaults.

### Step 3 — Solver (`solver.py`)

- **UnionFind** with path compression and union-by-size for near-constant-time cluster membership.
- **`gaec(instance)`** — Greedy Additive Edge Contraction over the **entire graph at once** (within-frame and between-frame edges compete in a single pool):
  1. Aggregate edge costs per cluster pair into `join_cost`.
  2. Push all positive pair-costs onto a max-heap (lazy deletion for stale entries).
  3. Repeatedly pop the highest-value pair; if still valid and positive, contract the two clusters and merge their cost tables.
  4. Stop when no positive join remains.

  Complexity **O(E log E)** — the full 145k-edge instance solves in a single pass with no decomposition.

- **Background handling:** most clutter points survive gating with no edges at all and naturally end up as singleton clusters — no special-case logic needed.
- **`greedy_solve(instance)`** wraps `gaec` and prints a run summary.
- **`variation_of_information(labels_true, labels_pred)`** computes VI from a contingency table: `VI = H(true) + H(pred) − 2·I(true; pred)`, in nats. VI = 0 means identical partitions; lower is better.

### Step 4 — Evaluation

Experiment harness sweeps **2 motion models × 4 observation-noise levels (0.05, 0.10, 0.20, 0.40) × 3 seeds**, reporting VI (primary, as assigned), with ARI and NMI as secondary metrics (`viz_report.partition_metrics`).

---

## 6. Results

| σ_obs | outcome                                                                                                   |
| ----- | --------------------------------------------------------------------------------------------------------- |
| 0.05  | **VI = 0.0** — perfect recovery of all four tracks (both motion models)                                   |
| 0.10  | VI ≈ 0.0 — essentially perfect                                                                            |
| 0.20  | partial failure — tracks begin to fragment/merge                                                          |
| 0.40  | **collapse: VI ≈ ln(4) ≈ 1.386** — auto-scaled gates grow so large that all points merge into one cluster |

The collapse value is an analytic anchor: merging everything into a single cluster against a balanced 4-way ground truth yields exactly `VI = ln 4`, which the sweep reproduces — a strong sanity check on both the pipeline and the metric implementation.

**Known limitation:** ~3.4% of within-frame edges connect points from different spheres or background (mean cost 0.13 vs. 0.228 for true same-sphere edges). At low noise GAEC's greedy ordering absorbs the strong true edges first, so these weak spurious edges rarely cause errors; at high noise they contribute to collapse.

---

## 7. Visualization & reporting (`viz_report.py`, `app.py`)

Report-grade figures used in the final talk:

- **Noise-degradation curve** — VI (and ARI/NMI) vs. σ_obs, one line per motion model, seed-based error bands.
- **Contingency heatmap** — true sphere labels vs. predicted clusters.
- **Before/after panel** — raw noisy detections vs. points colored by recovered track with start/end markers.
- **Edge-cost histograms** — same-sphere vs. cross-sphere edge cost distributions.
- **Collapse small-multiples** — predictions at low/mid/high noise side by side.
- **Gated-graph frame view** — nodes and gated edges of `G_t`, `G_{t+1}` and the between-frame edge set.
- **Scalability plot** — runtime vs. |E|, with the reference ILP's decomposition ceiling marked.

Deliberately **not** included: interpolated "recovered trajectory" lines (GAEC clusters observed points; it does not interpolate positions) and energy-conservation plots under nonzero motion noise (energy is not conserved by construction).

---

## 8. Design decisions

**GAEC instead of ILP (Gurobi).** With Prof. Andres's approval, the two-person team scope replaces the exact minimum-cost-multicut ILP with the GAEC heuristic. This turned out to be a substantive strength, not just a simplification: the reference Gurobi setup was limited by its trial license to sub-200-variable subproblems and had to decompose the instance, weakening any global-optimality claim. GAEC processes the full ~7,800-node / ~145k-edge graph **natively in one pass**, and at realistic noise levels reaches VI = 0 anyway.

**Unified single-pass clustering.** An earlier two-stage design (cluster per frame, then link clusters across frames) was replaced by one GAEC over all edges simultaneously, matching the updated project guide and removing an artificial stage boundary.

**Gating as implicit cut costs.** Filtering pairs with `d ≥ ρ` before cost computation keeps the edge set sparse and all costs positive; "do not join" is encoded by missing edges.

**Auto-gates.** Defaults `ρ_in = 4σ_obs + 0.05` and `ρ_mot = v·dt + 4σ_obs + 0.1` tie the gates to the noise model and the maximum plausible per-frame displacement (adjustable in the Streamlit app).

---

## 9. Reproducibility

- Single-seed RNG (`np.random.default_rng(seed)`) flows through simulation and observation.
- Experiment sweeps run over fixed seed sets {0, 1, 2}.
- Datasets round-trip via `save_dataset` / `load_dataset` (`.npz`, config included).

---

## 10. Acknowledgements

- **Prof. Bjoern Andres** — supervision, project definition, scope-reduction approval.
- **David Stein** — consulting throughout the project.
- Course: _Machine Learning for Computer Vision_, TU Dresden. Final talks: July 14/21, 2026.

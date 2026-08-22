# Tracking-by-Clustering (TBC)

Multi-object tracking on a 2-torus, formulated as a single **multicut** problem over a
spacetime graph and solved with **Greedy Additive Edge Contraction (GAEC)**.

There is no detection step, no data-association step and no per-frame assignment. Every
observed point in every frame is one node of one graph; a track is one connected component
of the partition that GAEC returns. Because the whole video is clustered at once, track
identities are stable across time by construction.

Course project for _Machine Learning for Computer Vision_ (MLCV), TU Dresden,
Chair of Machine Learning for Computer Vision (Prof. Dr. Björn Andres, consultant David Stein).
Authors: Daniel Mon (Part 1: world simulation, observation model, instance construction)
and Taha (Part 2: GAEC solver, constraint handling, evaluation, Streamlit app).

## Pipeline

```
SimConfig
   │
   ├─ simulate()            K spheres, elastic collisions, wrapped motion on [0,L)^2
   │        ↓  Trajectory (T, K, 2)
   ├─ sample_observations()  n_inliers per sphere + n_background clutter per frame
   │        ↓  points (M,2), times (M,), labels (M,)   [labels are sealed: evaluation only]
   ├─ build_instance()       gates + costs  →  edges (E,2), costs (E,), kind (E,)
   │        ↓  TrackingInstance
   ├─ greedy_solve()         GAEC on the full spacetime graph
   │        ↓  labels_pred (M,), objective
   └─ variation_of_information() / partition_metrics()
```

## Repository layout

```
.
├── app.py                  Streamlit explorer (run from the project root)
├── requirements.txt
├── test_suite.ipynb        notebook harness: sweeps, tables, report-grade numbers
├── tbc/
│   ├── __init__.py
│   ├── geometry.py         wrap, minimum-image displacement, torus distance
│   ├── synthesis.py        SimConfig, SyntheticDataset, generate/save/load
│   ├── motion.py           initial sampling, Euler step, simulate()
│   ├── collision.py        pairwise elastic collision resolution
│   ├── observation.py      inliers + uniform background clutter
│   ├── instance.py         gates, costs, TrackingInstance assembly
│   ├── solver.py           union-find, GAEC, greedy_solve, VI
│   ├── viz.py              COLORS, animate_world (app visuals)
│   └── viz_report.py       metrics + figure builders shared by app and notebook
└── report/
    ├── build.sh            pandoc + pandoc-crossref + citeproc
    ├── style.docx          reference template
    ├── refs.bib
    ├── sections/06-*.md … 11-*.md
    ├── figs/
    └── screenshots/
```

All modules import each other as `tbc.<module>`, so the package folder must be named `tbc/`
and commands must be run from the parent directory.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is required (`instance.py` uses `int | None`).

Interactive explorer:

```bash
streamlit run app.py
```

Minimal programmatic run:

```python
from tbc.synthesis import SimConfig, generate_dataset
from tbc.instance  import build_instance
from tbc.solver    import greedy_solve, variation_of_information

ds     = generate_dataset(SimConfig())      # 7,800 nodes at default settings
inst   = build_instance(ds)                 # prints a gate/edge summary
labels, objective = greedy_solve(inst)
print(variation_of_information(ds.labels, labels))
```

## Simulation defaults (`SimConfig`)

| Field                  | Default | Meaning                                                            |
| ---------------------- | ------- | ------------------------------------------------------------------ |
| `cube_size`            | 10.0    | side length `L` of the torus                                       |
| `n_spheres`            | 4       | number of objects `K`                                              |
| `radius`               | 0.5     | sphere radius, used for collisions and initial spacing             |
| `elasticity`           | 1.0     | restitution coefficient                                            |
| `speed`                | 1.0     | initial velocity magnitude                                         |
| `motion_noise_std`     | 0.05    | per-step Gaussian velocity perturbation (0 gives ballistic motion) |
| `n_timesteps`          | 60      | number of frames `T`                                               |
| `dt`                   | 0.1     | Euler step size                                                    |
| `n_inliers_per_sphere` | 20      | observed points per sphere per frame                               |
| `n_background`         | 50      | uniform clutter points per frame                                   |
| `obs_noise_std`        | 0.1     | Gaussian spread of inliers around the centre                       |
| `seed`                 | 0       | RNG seed                                                           |

Default graph size: `T · (K · n_inliers + n_background) = 60 · 130 = 7,800` nodes and roughly
125,000 candidate edges. The app starts from lighter settings (`T = 40`, 15 inliers, 30
background points) so that the browser stays responsive.

## How the instance is built

Everything is measured with the torus metric, so an object crossing a boundary is not torn
apart. `scipy.spatial.cKDTree(pos, boxsize=L)` returns minimum-image distances directly,
which avoids materialising the `M × M` matrix for the neighbour statistics.

**Two radii, not one.** Both are estimated from the data as `ρ = med(D) + c · MAD(D)`, where
`D` is the sample of first nearest-neighbour distances (within-frame for the spatial gate,
frame `t` to `t+1` for the motion gate). The median absolute deviation is used instead of the
standard deviation because clutter inflates the standard deviation, so a fixed `k · std` gate
grows together with the noise it is supposed to reject.

- `ρ_gate = med + c_gate · MAD` with `c_gate = 3.0` decides which edges enter the graph. It is
  deliberately generous, because a true edge that the gate drops can never be recovered (recall).
- `ρ_cost = med + c_cost · MAD` with `c_cost = 2.0` is the zero-crossing of the cost (precision).

The invariant `c_cost <= c_gate` is asserted in `build_instance`. It is the reason the problem
is a genuine multicut: edges with `ρ_cost < d < ρ_gate` are admitted with **negative** cost, so
repulsion is explicit rather than implied by the absence of an edge. Depending on noise level,
11% to 32% of admitted edges carry negative cost, rising monotonically with `obs_noise_std`.
Without that band GAEC would degenerate to connected components; with it, it does not.

**Cost.** `c_e = α · (ρ_cost − d)`, linear in the distance, positive inside `ρ_cost`, negative
outside, zero on the boundary. `α_in` and `α_mot` default to 1.0.

**Core-point pre-filter.** `core_point_mask` keeps only points with at least
`min_neighbors = 3` neighbours inside `ρ_gate` in their own frame, following the same
core/noise logic as DBSCAN. Filtered points are excluded from edge construction but stay in
the node array as singletons, so `labels_pred` remains index-aligned with `ds.labels` for
evaluation. Set `min_neighbors=None` to disable.

**Temporal wrap.** `between_frame_edges(..., cyclic=True)` also links frame `T-1` back to frame
`0`, treating time as the third periodic dimension. Note that the gate estimator
`nn_distances_inter` deliberately excludes the wrap pair from its sample, since the simulation
runs forward with no temporal periodicity and that pair is not a one-step displacement.

## The solver

`gaec(inst)` contracts edges greedily, always taking the currently largest positive joined
cost, and stops when no positive edge remains.

- **Max-heap with lazy deletion** instead of an `O(E²)` rescan. Contraction changes the cost of
  many pairs at once, so stale entries are left in the heap and validated on pop.
- **Four-part staleness protocol** on each pop: the popped value must be positive, the two
  endpoints must not already share a root, the merged key must still exist in `join`, and its
  stored value must match the popped one to within `1e-9`. A surviving key whose value has
  changed is pushed back with its current value rather than discarded. Stale entries can both
  overstate and understate the true cost, which is why the value check is two-sided rather than
  a simple "skip if smaller".
- **Union-find** with union by size and **path halving** (`parent[i] = parent[parent[i]]`).
  The docstring in `solver.py` says "path compression"; the implementation is halving, which
  gives the same near-constant amortised behaviour without recursion.
- **Adjacency folding** touches only the neighbours of the absorbed root, so the work per
  contraction is proportional to that root's degree, not to `E`.
- Output roots are relabelled to consecutive ids via `np.unique(..., return_inverse=True)`.

`greedy_solve` wraps this and returns `(labels, objective)`, where the objective is the sum of
costs of all edges whose endpoints ended up in the same component, and prints a short summary
(cluster count, clusters of size >= 10, singletons).

Measured runtime is 0.6 s to 1.1 s at 7,800 nodes, well below any ILP-based baseline on the
same instance.

## Evaluation

`variation_of_information(true, pred)` in `solver.py` returns
`VI = H(P) + H(T) − 2·I(P;T)` in nats. `partition_metrics` in `viz_report.py` returns `vi`,
`ari` and `nmi` from a single contingency table.

VI decomposes into `H(P|T)` (split error, one true track scattered over several clusters) and
`H(T|P)` (merge error, several true tracks fused into one). For `K = 4`, total collapse to a
single cluster gives `ln(4) ≈ 1.386`.

Observed behaviour on the default configuration:

- Near-exact recovery up to `obs_noise_std = 0.20`.
- Breakdown occurs in the `0.20 → 0.40` window. At `σ = 0.40`, VI reaches about 1.5, which is
  **above** `ln(4)`, so the failure is mixed (splits and merges together), not pure fusion.
- `H(T|P) = 0` for `σ <= 0.10`: no merges at all in that regime.
- `H(P|T)` is nonzero throughout, because the `min_neighbors = 3` filter isolates a few inlier
  points as singletons.
- A connected-components baseline is competitive with, and in several grid cells better than,
  GAEC at low to moderate noise. GAEC's advantage has to be argued on the repulsive-edge regime,
  not asserted globally.
- Removing background clutter at moderate noise makes VI worse, not better. Clutter raises the
  median nearest-neighbour distance and therefore the estimated gates; without it the gates
  shrink and true tracks fragment.

The app computes metrics on inliers only (`ds.labels >= 0`); the notebook harness produces the
full report-grade tables.

## Streamlit explorer (`app.py`)

Five sections, all driven by the sidebar and cached per parameter tuple with `st.cache_data`:

1. **Ground-truth world**, animated 2D view of the `K` spheres.
2. **Point cloud frame by frame**, radar view (what the tracker sees) next to the truth view.
   The two panels have **independent** frame sliders.
3. **Instance graph at (t, t+1)**, within-frame and between-frame edges drawn in 3D with the
   two frame planes.
4. **GAEC result**, points coloured by predicted track with a circle around each recovered
   track, plus the truth-versus-prediction animation.
5. **Performance**, in three tabs: _this run_ (cluster size distribution on a log axis and the
   truth-versus-cluster contingency heatmap), _edge quality_ (edge-cost histograms split by the
   sealed labels, within-frame and between-frame), and _sensitivity (live)_ (a reduced mini
   sweep over `n_background`, `obs_noise_std`, `motion_noise_std`, `n_spheres` or `speed` at
   `T = 20` with two seeds).

The contingency heatmap is a permutation matrix by construction, since clusters are relabelled
by first appearance; it is not "near-diagonal" in any informative sense.

## Known differences and caveats

- **`cyclic` differs between entry points.** `build_instance` defaults to `cyclic=True` and the
  notebook harness uses that default; `app.py` passes `cyclic=False`. This is a real
  methodological difference between the reported numbers and the live demo, and it is stated
  explicitly in report sections 9.2 and 10.1 rather than smoothed over.
- Metrics in the app are restricted to inliers; background points are excluded from VI there.
- `min_neighbors = 3` guarantees a nonzero split error even in the easy regime. Lowering it
  raises recall at the cost of noise bridges between objects.
- The symbol `T` is used for both the frame count and the true partition in the report. Keep the
  two usages separated when reading Sections 6 to 11.

## Reproducing the report figures

```bash
cd report
./build.sh          # pandoc + pandoc-crossref + citeproc + style.docx -> Part2.docx
```

`Part2.docx` is disposable build output. Edit `report/sections/0N-*.md`, never the `.docx`.
Before building, catch duplicate cross-reference labels, which abort pandoc-crossref hard:

```bash
grep -oh '{#[a-z]*:[a-zA-Z0-9_-]*}' sections/*.md | sort | uniq -d
```

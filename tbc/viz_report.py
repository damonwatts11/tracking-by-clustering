"""
viz_report.py -- presentation visuals for the project hearing.

Implements the Tier A/B/C visuals from VIZ_TODO:
  - before_after_panel        (Tier A: raw cloud vs recovered tracks)
  - contingency_heatmap       (Tier A: which spheres merged/split)
  - noise_curve               (Tier A: VI/ARI/NMI vs obs noise, per model)
  - gated_frame_view          (Tier C: one frame's gated graph, made concrete)
  - edge_cost_histograms      (Tier B: same-sphere vs other edge costs)
  - collapse_small_multiples  (Tier B: predicted labels at low/mid/high noise)
  - scalability_plot          (Tier B: runtime vs graph size, Gurobi ceiling line)

Plus pure-numpy partition metrics (VI, ARI, NMI) in one call.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tbc.viz import COLORS, _nan_break

GRAY = "#B8B8B8"


# ------------------------------------------------------------------
# Metrics: VI + ARI + NMI from one contingency table (numpy only)
# ------------------------------------------------------------------

def contingency_table(labels_true, labels_pred):
    """Counts N[i, j] = #points with true id i and predicted id j.
    Also returns the sorted unique ids for rows/cols."""
    tu, ti = np.unique(labels_true, return_inverse=True)
    pu, pi = np.unique(labels_pred, return_inverse=True)
    N = np.zeros((len(tu), len(pu)))
    np.add.at(N, (ti, pi), 1.0)
    return N, tu, pu


def partition_metrics(labels_true, labels_pred):
    """Return dict with vi (nats), ari, nmi for two labelings."""
    N, _, _ = contingency_table(labels_true, labels_pred)
    n = N.sum()
    r = N / n
    p = r.sum(axis=1)
    q = r.sum(axis=0)

    def H(x):
        x = x[x > 0]
        return float(-(x * np.log(x)).sum())

    mask = r > 0
    I = float((r[mask] * np.log(r[mask] / np.outer(p, q)[mask])).sum())
    vi = max(H(p) + H(q) - 2 * I, 0.0)

    # NMI (arithmetic normalisation)
    denom = 0.5 * (H(p) + H(q))
    nmi = I / denom if denom > 0 else 1.0

    # ARI from pair counts
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_ij = comb2(N).sum()
    a = comb2(N.sum(axis=1)).sum()
    b = comb2(N.sum(axis=0)).sum()
    total = comb2(n)
    expected = a * b / total if total > 0 else 0.0
    max_index = 0.5 * (a + b)
    ari = ((sum_ij - expected) / (max_index - expected)
           if max_index != expected else 1.0)

    return {"vi": vi, "ari": float(ari), "nmi": float(nmi)}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _circular_mean(x, L):
    """Mean of positions on a circle of circumference L (per column)."""
    theta = x * (2 * np.pi / L)
    m = np.arctan2(np.sin(theta).mean(axis=0), np.cos(theta).mean(axis=0))
    return np.mod(m * (L / (2 * np.pi)), L)


def track_centroids(points, times, labels_pred, track_id, T, L):
    """Per-frame circular-mean centroid of one predicted track.
    Returns (frames_present, centroids (F,2))."""
    mask = labels_pred == track_id
    fr, cent = [], []
    tt = times[mask]
    pp = points[mask]
    for t in np.unique(tt):
        fr.append(int(t))
        cent.append(_circular_mean(pp[tt == t], L))
    return np.array(fr), np.array(cent)


def large_tracks(labels_pred, min_size=10):
    """Predicted track ids with at least min_size members, largest first."""
    ids, counts = np.unique(labels_pred, return_counts=True)
    keep = ids[counts >= min_size]
    order = np.argsort(-counts[counts >= min_size])
    return keep[order]


# ------------------------------------------------------------------
# Tier A -- Before / After panel
# ------------------------------------------------------------------

def before_after_panel(ds, labels_pred, min_track_size=10):
    """Left: raw detections, no identity. Right: points coloured by
    predicted track, solid centroid trajectory per large track.
    No interpolation of missing frames -- lines pass only through
    frames where the cluster actually has points."""
    cfg = ds.config
    L, T, dt = cfg.cube_size, cfg.n_timesteps, cfg.dt

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Input: anonymous point cloud",
                        "Output: recovered tracks (GAEC)"),
        horizontal_spacing=0.02,
    )

    # LEFT -- everything gray, no identity
    fig.add_trace(go.Scatter3d(
        x=ds.points[:, 0], y=ds.points[:, 1], z=ds.times * dt,
        mode="markers", name="detections",
        marker=dict(size=1.6, color=GRAY, opacity=0.5),
        showlegend=False,
    ), row=1, col=1)

    # RIGHT -- coloured by predicted track
    tracks = large_tracks(labels_pred, min_track_size)
    small_mask = ~np.isin(labels_pred, tracks)
    fig.add_trace(go.Scatter3d(
        x=ds.points[small_mask, 0], y=ds.points[small_mask, 1],
        z=ds.times[small_mask] * dt,
        mode="markers", name="clutter (small clusters)",
        marker=dict(size=1.2, color=GRAY, opacity=0.25),
    ), row=1, col=2)

    for i, tid in enumerate(tracks):
        color = COLORS[i % len(COLORS)]
        m = labels_pred == tid
        fig.add_trace(go.Scatter3d(
            x=ds.points[m, 0], y=ds.points[m, 1], z=ds.times[m] * dt,
            mode="markers", name=f"track {i}",
            marker=dict(size=1.8, color=color, opacity=0.55),
        ), row=1, col=2)

        # solid centroid trajectory through the real clustered nodes only
        fr, cent = track_centroids(ds.points, ds.times, labels_pred,
                                   tid, T, L)
        broken = _nan_break(cent, L)
        z = np.linspace(fr[0] * dt, fr[-1] * dt, len(broken))
        fig.add_trace(go.Scatter3d(
            x=broken[:, 0], y=broken[:, 1], z=z,
            mode="lines", showlegend=False,
            line=dict(color=color, width=5),
        ), row=1, col=2)

        # start / end markers
        fig.add_trace(go.Scatter3d(
            x=[cent[0, 0], cent[-1, 0]], y=[cent[0, 1], cent[-1, 1]],
            z=[fr[0] * dt, fr[-1] * dt],
            mode="markers", showlegend=False,
            marker=dict(size=[6, 6], color=color,
                        symbol=["circle", "diamond"]),
        ), row=1, col=2)

    scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="time",
                 aspectmode="cube",
                 xaxis=dict(range=[0, L]), yaxis=dict(range=[0, L]))
    fig.update_layout(scene=scene, scene2=scene,
                      height=550, margin=dict(l=0, r=0, t=60, b=0),
                      legend=dict(orientation="h", y=-0.05))
    return fig


# ------------------------------------------------------------------
# Tier A -- Contingency heatmap (inliers only)
# ------------------------------------------------------------------

def contingency_heatmap(labels_true, labels_pred, top=8):
    """Rows: true spheres. Cols: the `top` largest predicted clusters
    (by inlier count) + an 'other' bucket. Inliers only."""
    inl = labels_true >= 0
    N, tu, pu = contingency_table(labels_true[inl], labels_pred[inl])

    col_sizes = N.sum(axis=0)
    order = np.argsort(-col_sizes)
    keep = order[:top]
    rest = order[top:]

    M = N[:, keep]
    col_names = [f"cluster {int(pu[j])}" for j in keep]
    if len(rest) > 0:
        M = np.concatenate([M, N[:, rest].sum(axis=1, keepdims=True)], axis=1)
        col_names.append(f"other ({len(rest)})")
    row_names = [f"sphere {int(i)}" for i in tu]

    fig = go.Figure(go.Heatmap(
        z=M, x=col_names, y=row_names,
        colorscale="Blues", text=M.astype(int), texttemplate="%{text}",
        colorbar=dict(title="points"),
    ))
    fig.update_layout(
        title="Contingency: true spheres x predicted clusters (inliers only)",
        xaxis_title="predicted", yaxis_title="ground truth",
        yaxis=dict(autorange="reversed"),
        height=380, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ------------------------------------------------------------------
# Tier C -- Gated graph, one frame
# ------------------------------------------------------------------

def gated_frame_view(instance, frame_t, rho_in, L, show_between=True):
    """One frame's nodes + within-frame edges (+ edges to t+1 in a
    second colour). Gate radius rho_in drawn around one node.
    Edges that would cross the torus wrap are skipped for drawing
    clarity (they are still in the instance)."""
    pts, times = instance.points, instance.times
    e, k = instance.edges, instance.kind

    in_t = times == frame_t
    in_next = times == ((frame_t + 1) % instance.T)

    fig = go.Figure()

    def _segments(edge_rows, color, name):
        xs, ys = [], []
        for i, j in edge_rows:
            a, b = pts[i], pts[j]
            if np.any(np.abs(a - b) > L / 2):   # wrap edge, skip drawing
                continue
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                 line=dict(color=color, width=1),
                                 opacity=0.5))

    # within-frame edges of frame t
    wmask = (k == 0) & in_t[e[:, 0]] & in_t[e[:, 1]]
    _segments(e[wmask], "#2E86AB", f"within-frame E_{frame_t}")

    # between-frame edges t -> t+1
    if show_between:
        bmask = (k == 1) & ((in_t[e[:, 0]] & in_next[e[:, 1]]) |
                            (in_next[e[:, 0]] & in_t[e[:, 1]]))
        _segments(e[bmask], "#F4B942",
                  f"between-frame E_{frame_t},{(frame_t+1) % instance.T}")

    # nodes
    fig.add_trace(go.Scatter(
        x=pts[in_t, 0], y=pts[in_t, 1], mode="markers",
        name=f"frame {frame_t}", marker=dict(size=6, color="#333")))
    if show_between:
        fig.add_trace(go.Scatter(
            x=pts[in_next, 0], y=pts[in_next, 1], mode="markers",
            name=f"frame {(frame_t+1) % instance.T}",
            marker=dict(size=6, color="#999", symbol="circle-open")))

    # gate circle around one node (first node of frame t)
    c = pts[np.flatnonzero(in_t)[0]]
    th = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=c[0] + rho_in * np.cos(th), y=c[1] + rho_in * np.sin(th),
        mode="lines", name=f"gate ρ_in = {rho_in:.2f}",
        line=dict(color="#E94F37", dash="dash")))

    fig.update_layout(
        title=f"The gated graph, frame {frame_t} (wrap edges omitted from drawing)",
        xaxis=dict(range=[0, L], title="x", constrain="domain"),
        yaxis=dict(range=[0, L], title="y", scaleanchor="x"),
        width=620, height=620,
    )
    return fig


# ------------------------------------------------------------------
# Tier B -- Edge-cost histograms (uses sealed labels: debug only)
# ------------------------------------------------------------------

def edge_cost_histograms(instance, labels_gt, kind=0):
    """Overlaid histograms of edge costs, split by whether the edge is
    a true same-sphere pair. kind: 0 = within-frame, 1 = between-frame."""
    e = instance.edges
    m = instance.kind == kind
    li, lj = labels_gt[e[m, 0]], labels_gt[e[m, 1]]
    same = (li == lj) & (li >= 0)
    costs = instance.costs[m]

    wrong = float(((costs > 0) & ~same).sum()) / max((costs > 0).sum(), 1)

    name = "within-frame" if kind == 0 else "between-frame"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=costs[same], name="true same-sphere",
                               marker_color="#44BBA4", opacity=0.65))
    fig.add_trace(go.Histogram(x=costs[~same], name="different / background",
                               marker_color="#E94F37", opacity=0.65))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        barmode="overlay",
        title=(f"{name} edge costs -- {wrong:.1%} of positive edges are wrong "
               f"(labels used for this plot only)"),
        xaxis_title="cost c_e", yaxis_title="count", height=380,
    )
    return fig


# ------------------------------------------------------------------
# Tier A -- Noise degradation curve
# ------------------------------------------------------------------

def noise_curve(results, metric="vi", title=None):
    """results: list of dicts with keys model, noise, seed, vi, ari, nmi.
    Line per model, shaded ±1 std over seeds."""
    models = sorted({r["model"] for r in results})
    palette = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    fig = go.Figure()

    for m in models:
        noises = sorted({r["noise"] for r in results if r["model"] == m})
        mean, std = [], []
        for nz in noises:
            vals = [r[metric] for r in results
                    if r["model"] == m and r["noise"] == nz]
            mean.append(np.mean(vals))
            std.append(np.std(vals))
        mean, std = np.array(mean), np.array(std)
        c = palette[m]
        rgb = tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(
            x=noises + noises[::-1],
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself", fillcolor=f"rgba{rgb + (0.18,)}",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=noises, y=mean, mode="lines+markers", name=m,
            line=dict(color=c, width=3), marker=dict(size=8)))

    fig.update_layout(
        title=title or f"{metric.upper()} vs observation noise (±1 std over seeds)",
        xaxis_title="obs_noise_std", yaxis_title=metric.upper(),
        height=420,
    )
    return fig


# ------------------------------------------------------------------
# Tier B -- Collapse small multiples
# ------------------------------------------------------------------

def collapse_small_multiples(runs, frame_t, L, min_track_size=10):
    """runs: list of (noise_value, points, times, labels_pred, vi).
    One 2D scatter per noise level of frame `frame_t`, coloured by
    predicted track. Merging shows up as one colour swallowing spheres."""
    n = len(runs)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"σ={nz}  VI={vi:.2f}"
                                        for nz, _, _, _, vi in runs])
    for col, (nz, pts, tms, lab, vi) in enumerate(runs, start=1):
        m = tms == frame_t
        tracks = large_tracks(lab, min_track_size)
        color_of = {tid: COLORS[i % len(COLORS)]
                    for i, tid in enumerate(tracks)}
        cols = [color_of.get(l, GRAY) for l in lab[m]]
        fig.add_trace(go.Scatter(
            x=pts[m, 0], y=pts[m, 1], mode="markers",
            marker=dict(size=5, color=cols), showlegend=False,
        ), row=1, col=col)
        fig.update_xaxes(range=[0, L], row=1, col=col, constrain="domain")
        fig.update_yaxes(range=[0, L], row=1, col=col,
                         scaleanchor=f"x{col if col > 1 else ''}")
    fig.update_layout(title=f"Predicted tracks at frame {frame_t}, "
                            f"low → high noise (colour = one predicted cluster)",
                      height=380)
    return fig


# ------------------------------------------------------------------
# Tier B -- Scalability plot
# ------------------------------------------------------------------

def scalability_plot(n_edges, runtimes, gurobi_ceiling=200, label_x="edges |E|"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_edges, y=runtimes, mode="lines+markers", name="our GAEC",
        line=dict(color="#2E86AB", width=3), marker=dict(size=8)))
    fig.add_vline(x=gurobi_ceiling, line_dash="dash", line_color="#E94F37",
                  annotation_text=f"exact-solver ceiling (~{gurobi_ceiling} vars)",
                  annotation_position="top right")
    fig.update_layout(
        title="Runtime vs problem size -- one GAEC pass over the full graph",
        xaxis_title=label_x, yaxis_title="runtime [s]",
        xaxis_type="log", yaxis_type="log", height=420,
    )
    return fig

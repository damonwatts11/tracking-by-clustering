"""
app.py -- interactive Tracking-by-Clustering explorer (Streamlit).

Run from the project root (the folder containing tbc/):
    streamlit run app.py

Sections:
  1. Sidebar        -- every SimConfig parameter + gate controls
  2. World overview -- animated 2D view of the ground-truth spheres
  3. Radar vs truth -- two synced 2D views of the point cloud at frame t
  4. Instance graph -- within-frame + between-frame edges at (t, t+1)
  5. Tracker        -- GAEC result: points coloured by predicted track,
                       a circle ("box") around each recovered track at frame t
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from tbc.synthesis import SimConfig, generate_dataset
from tbc.instance import build_instance, estimate_gates_from_dataset
from tbc.solver import greedy_solve
from tbc.viz import COLORS, animate_world
from tbc.viz_report import (partition_metrics, large_tracks, _circular_mean,
                            contingency_heatmap, edge_cost_histograms,
                            noise_curve)
st.set_page_config(page_title="Tracking-by-Clustering explorer", layout="wide")

# ------------------------------------------------------------------
# Sidebar -- all parameters
# ------------------------------------------------------------------
sb = st.sidebar
sb.title("Simulation")
L      = sb.slider("world size L", 5.0, 20.0, 10.0, 1.0)
K      = sb.slider("number of spheres K", 1, 8, 4)
radius = sb.slider("sphere radius r", 0.2, 1.5, 0.5, 0.1)
speed  = sb.slider("speed", 0.2, 3.0, 1.0, 0.1)
mnoise = sb.select_slider("motion model",
                          options=[0.0, 0.02, 0.05, 0.1, 0.2],
                          value=0.05,
                          format_func=lambda v: "ballistic (0)" if v == 0 else f"random walk σ={v}")
T      = sb.slider("frames T", 10, 100, 40, 5)
dt     = 0.1

sb.title("Observation model")
nin    = sb.slider("inliers per sphere / frame", 3, 40, 15)
nbg    = sb.slider("background points / frame", 0, 150, 30, 5)
onoise = sb.select_slider("observation noise σ_obs",
                          options=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4], value=0.1)
seed   = sb.number_input("seed", 0, 999, 0)

sb.title("Instance (gates)")
c_gate = sb.slider("c_gate (edge admission, recall)", 1.0, 6.0, 3.0, 0.5,
                   help="ρ_gate = med + c_gate·MAD. Generous: the gate must "
                        "not drop true edges.")
c_cost = sb.slider("c_cost (cost zero-crossing)", 0.0, 3.0, 2.0, 0.25,
                   help="ρ_cost = med + c_cost·MAD. Edges with "
                        "ρ_cost < d < ρ_gate get negative (repulsive) cost.")
min_nb = sb.slider("min neighbours (noise pre-filter)", 0, 8, 3,
                   help="Points with fewer neighbours inside ρ_gate in their "
                        "own frame are excluded from the graph. 0 = off.")
c_cost = min(c_cost, c_gate)
gate_key = (c_gate, c_cost, min_nb)

run_tracker = sb.toggle("run GAEC tracker", value=False)

params = (L, K, radius, speed, mnoise, T, dt, nin, nbg, onoise, int(seed))


# ------------------------------------------------------------------
# Cached pipeline stages
# ------------------------------------------------------------------
@st.cache_data(show_spinner="simulating world + sampling observations ...")
def get_dataset(p):
    L, K, r, sp, mn, T, dt, nin, nbg, on, seed = p
    cfg = SimConfig(cube_size=L, n_spheres=K, radius=r, elasticity=1.0,
                    speed=sp, motion_noise_std=mn, n_timesteps=T, dt=dt,
                    n_inliers_per_sphere=nin, n_background=nbg,
                    obs_noise_std=on, seed=seed)
    return generate_dataset(cfg)

@st.cache_data(show_spinner="building the instance graph ...")
def get_instance(p, gkey):
    ds = get_dataset(p)
    cg, cc, mn = gkey
    return build_instance(
        ds, c_gate_spatial=cg, c_gate_motion=cg,
        c_cost_spatial=cc, c_cost_motion=cc,
        min_neighbors=(mn if mn > 0 else None),
        cyclic=False, verbose=False)


@st.cache_data(show_spinner="estimating gates from the cloud ...")
def get_estimated_gates(p, cg):
    ds = get_dataset(p)
    return estimate_gates_from_dataset(ds, c_spatial=cg, c_motion=cg)


@st.cache_data(show_spinner="running GAEC on the full spacetime graph ...")
def get_solution(p, gkey):
    inst = get_instance(p, gkey)
    ds = get_dataset(p)
    t0 = time.perf_counter()
    labels, objective = greedy_solve(inst)
    rt = time.perf_counter() - t0
    inl = ds.labels >= 0
    met = partition_metrics(ds.labels[inl], labels[inl])
    return labels, objective, rt, met


ds = get_dataset(params)
inst = get_instance(params, gate_key)
cfg = ds.config

_rs, _rm = get_estimated_gates(params, c_gate)
sb.caption(f"estimated from data:  ρ_gate,in = {_rs:.3f}   ρ_gate,mot = {_rm:.3f}")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Tracking-by-Clustering — interactive explorer")
c1, c2, c3, c4 = st.columns(4)
c1.metric("points M", len(ds.points))
c2.metric("frames T", T)
c3.metric("within-frame edges", int((inst.kind == 0).sum()))
c4.metric("between-frame edges", int((inst.kind == 1).sum()))

# ------------------------------------------------------------------
# 1 · World overview animation
# ------------------------------------------------------------------
st.header("1 · Ground-truth world (animated)")
st.caption("The hidden truth: K sphere centers drifting, colliding, wrapping. "
           "Press ▶.")
st.plotly_chart(animate_world(ds.trajectory, cfg), width='content')

# ------------------------------------------------------------------
# Shared frame slider drives every panel below
# ------------------------------------------------------------------
st.header("2 · The point cloud, frame by frame")

def _square_layout(fig, title):
    fig.update_layout(
        title=title, width=520, height=520, showlegend=False,
        xaxis=dict(range=[0, L], constrain="domain", title="x"),
        yaxis=dict(range=[0, L], scaleanchor="x", title="y"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _add_player(fig, T, title):
    """Attach play/pause buttons + a frame slider to an animated figure."""
    fig = _square_layout(fig, title)
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.15, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(f)], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))],
                        label=str(f)) for f in range(T)],
            x=0, y=-0.12, len=1.0,
        )],
    )
    return fig


left, right = st.columns(2)

with left:
    radar_frames = []
    for f in range(T):
        m = ds.times == f
        radar_frames.append(go.Frame(
            data=[go.Scatter(x=ds.points[m, 0], y=ds.points[m, 1],
                             mode="markers",
                             marker=dict(size=6, color="white"))],
            name=str(f)))
    fig = go.Figure(data=radar_frames[0].data, frames=radar_frames)
    st.plotly_chart(_add_player(fig, T, "What the radar sees (anonymous)"))

with right:
    truth_frames = []
    for f in range(T):
        m = ds.times == f
        data = [go.Scatter(x=ds.points[m & (ds.labels == -1), 0],
                           y=ds.points[m & (ds.labels == -1), 1],
                           mode="markers",
                           marker=dict(size=6, color="white",
                                       line=dict(color="black", width=1)))]
        for k in range(K):
            mk = m & (ds.labels == k)
            data.append(go.Scatter(x=ds.points[mk, 0], y=ds.points[mk, 1],
                                   mode="markers",
                                   marker=dict(size=6,
                                               color=COLORS[k % len(COLORS)])))
        truth_frames.append(go.Frame(data=data, name=str(f)))
    fig = go.Figure(data=truth_frames[0].data, frames=truth_frames)
    st.plotly_chart(_add_player(fig, T, "Ground truth (sealed labels)"))

# ------------------------------------------------------------------
# 3 · The instance graph in 3D — two stacked frame planes
# ------------------------------------------------------------------
st.header("3 · The instance graph at (t, t+1)")
t = st.slider("frame t — choose the frame pair to inspect", 0, T - 1, 0)
t_next = (t + 1) % T
mask_t = ds.times == t
mask_n = ds.times == t_next

st.caption("The lecture picture, live: frame t is the lower plane, frame t+1 "
           "the upper one. Light blue = within-frame edges E_t and E_(t+1). "
           "Orange = between-frame edges E_t,t+1 spanning the planes. "
           "Rotate with the mouse. Wrap-crossing edges are kept in the "
           "instance but omitted from the drawing.")

MAX_SEG = 3000
e, kd = inst.edges, inst.kind
w_t = (kd == 0) & mask_t[e[:, 0]] & mask_t[e[:, 1]]
w_n = (kd == 0) & mask_n[e[:, 0]] & mask_n[e[:, 1]]
btw = (kd == 1) & ((mask_t[e[:, 0]] & mask_n[e[:, 1]]) |
                   (mask_n[e[:, 0]] & mask_t[e[:, 1]]))

zlev = np.zeros(len(ds.points))
zlev[mask_n] = 1.0


def _seg3d(rows, color, name, width=2,opacity=0.45):
    if len(rows) > MAX_SEG:
        rows = rows[np.random.default_rng(0).choice(len(rows), MAX_SEG,
                                                    replace=False)]
    xs, ys, zs = [], [], []
    for i, j in rows:
        a, bb = ds.points[i], ds.points[j]
        if np.any(np.abs(a - bb) > L / 2):
            continue
        xs += [a[0], bb[0], None]
        ys += [a[1], bb[1], None]
        zs += [zlev[i], zlev[j], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                        opacity=opacity, line=dict(color=color, width=width))


def _plane(z, color="gray"):
    return go.Scatter3d(x=[0, L, L, 0, 0], y=[0, 0, L, L, 0],
                        z=[z] * 5, mode="lines", showlegend=False,
                        line=dict(color=color, width=2))


fig = go.Figure()
fig.add_trace(_plane(0.0))
fig.add_trace(_plane(1.0))
fig.add_trace(_seg3d(e[btw], "#808080", f"E_{t},{t_next} (between)",
                     width=0.5, opacity=0.15))
fig.add_trace(_seg3d(e[w_t], "#00E5FF", f"E_{t} (within)",
                     width=2, opacity=0.9))
fig.add_trace(_seg3d(e[w_n], "#FFD166", f"E_{t_next} (within)",
                     width=2, opacity=0.9))
fig.add_trace(go.Scatter3d(
    x=ds.points[mask_t, 0], y=ds.points[mask_t, 1], z=zlev[mask_t],
    mode="markers", name=f"frame {t}",
    marker=dict(size=3.5, color="#0095FF")))
fig.add_trace(go.Scatter3d(
    x=ds.points[mask_n, 0], y=ds.points[mask_n, 1], z=zlev[mask_n],
    mode="markers", name=f"frame {t_next}",
    marker=dict(size=3.5, color="#FF8B28")))

fig.update_layout(
    title=f"G_t and G_(t+1) with E_t,t+1 — frames {t} and {t_next}",
    height=700, margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(
        xaxis=dict(range=[0, L], title="x"),
        yaxis=dict(range=[0, L], title="y"),
        zaxis=dict(range=[-0.15, 1.15], tickvals=[0, 1],
                   ticktext=[f"frame {t}", f"frame {t_next}"], title=""),
        aspectmode="cube",
    ),
)
if int(w_t.sum()) > MAX_SEG or int(btw.sum()) > MAX_SEG:
    st.caption(f"(showing a random {MAX_SEG}-edge subsample per type for speed)")
st.plotly_chart(fig)

# ------------------------------------------------------------------
# 4 · The tracker
# ------------------------------------------------------------------
st.header("4 · GAEC result")
if not run_tracker:
    st.info("Toggle **run GAEC tracker** in the sidebar. First run takes a few "
            "seconds to minutes depending on size; results are cached.")
else:
    labels_pred, objective, rt, met = get_solution(params, gate_key)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VI (inliers)", f"{met['vi']:.3f}")
    m2.metric("ARI", f"{met['ari']:.3f}")
    m3.metric("clusters ≥ 10 pts",
              int((np.unique(labels_pred, return_counts=True)[1] >= 10).sum()))
    m4.metric("solve time", f"{rt:.1f} s")

    tracks = large_tracks(labels_pred, min_size=10)
    color_of = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(tracks)}

  # ---- shared helpers for both animated panels ----
    def _circle_pts(c, r, n=40):
        th = np.linspace(0, 2 * np.pi, n)
        return c[0] + r * np.cos(th), c[1] + r * np.sin(th)

    cent = {}
    for tid in tracks:
        mtid = labels_pred == tid
        tt, pp = ds.times[mtid], ds.points[mtid]
        for f in np.unique(tt):
            cent[(tid, int(f))] = _circular_mean(pp[tt == f], L)

    small_all = ~np.isin(labels_pred, tracks)
    _empty = go.Scatter(x=[], y=[], mode="lines", showlegend=False)

    colA, colB = st.columns(2)

    # ---- panel A: point cloud coloured by predicted track (animated) ----
    with colA:
        cloud_frames = []
        for f in range(T):
            mf = ds.times == f
            data = [go.Scatter(
                x=ds.points[mf & small_all, 0],
                y=ds.points[mf & small_all, 1],
                mode="markers", name="clutter",
                marker=dict(size=5, color="gray"))]
            for i, tid in enumerate(tracks):
                mk = mf & (labels_pred == tid)
                data.append(go.Scatter(
                    x=ds.points[mk, 0], y=ds.points[mk, 1],
                    mode="markers", name=f"track {i}",
                    marker=dict(size=6, color=color_of[tid])))
            for tid in tracks:                      # tracking circles
                c = cent.get((tid, f))
                if c is None:
                    data.append(_empty)
                else:
                    cx, cy = _circle_pts(c, 1.3 * radius)
                    data.append(go.Scatter(
                        x=cx, y=cy, mode="lines", showlegend=False,
                        line=dict(color=color_of[tid], width=3)))
            cloud_frames.append(go.Frame(data=data, name=str(f)))

        figP = go.Figure(data=cloud_frames[0].data, frames=cloud_frames)
        st.plotly_chart(_add_player(figP, T,
                        "Predicted tracks on the point cloud (animated)"))

    # ---- panel B: true spheres vs GAEC circles (animated) ----
    with colB:
        anim_frames = []
        for f in range(T):
            data = []
            for k in range(K):
                cx, cy = _circle_pts(ds.trajectory.centers[f, k], radius)
                data.append(go.Scatter(
                    x=cx, y=cy, mode="lines", showlegend=False,
                    line=dict(color="lightgray", width=1.5),
                    fill="toself", fillcolor="rgba(255,255,255,0.12)"))
            for tid in tracks:
                c = cent.get((tid, f))
                if c is None:
                    data.append(_empty)
                else:
                    cx, cy = _circle_pts(c, 1.3 * radius)
                    data.append(go.Scatter(
                        x=cx, y=cy, mode="lines", showlegend=False,
                        line=dict(color=color_of[tid], width=3)))
            anim_frames.append(go.Frame(data=data, name=str(f)))

        figA = go.Figure(data=anim_frames[0].data, frames=anim_frames)
        st.plotly_chart(_add_player(figA, T,
                        "True spheres vs GAEC tracks (animated)"))

    st.caption("Same frame slider as above: scrub time and watch the circles "
               "follow the spheres. Colours are stable across frames because a "
               "track is ONE cluster of the whole spacetime graph.")
# ------------------------------------------------------------------
# 5 · Performance
# ------------------------------------------------------------------
st.header("5 · Performance")
tab1, tab2, tab3 = st.tabs(["this run", "edge quality", "sensitivity (live)"])

with tab1:
    if not run_tracker:
        st.info("Run the GAEC tracker (sidebar toggle) to see per-run metrics.")
    else:
        labels_pred, objective, rt, met = get_solution(params, gate_key)

        st.subheader("Cluster size distribution")
        _, counts = np.unique(labels_pred, return_counts=True)
        counts = np.sort(counts)[::-1][:40]
        figc = go.Figure(go.Bar(y=counts, marker_color="#4FC3F7"))
        figc.update_layout(height=320, yaxis_type="log",
                           xaxis_title="cluster rank",
                           yaxis_title="size (log)",
                           title=f"K = {K} tall towers = tracks; "
                                 "the tail = background clutter")
        st.plotly_chart(figc, width='stretch')

        st.subheader("Which sphere went to which cluster?")
        st.plotly_chart(contingency_heatmap(ds.labels, labels_pred, top=8),
                        width='stretch')

with tab2:
    st.caption("Edge costs split by the sealed labels — diagnostic only, "
               "never seen by the tracker. Overlap across zero = the wrong "
               "edges the solver must out-vote.")
    st.plotly_chart(edge_cost_histograms(inst, ds.labels, kind=0),
                    width='stretch')
    st.plotly_chart(edge_cost_histograms(inst, ds.labels, kind=1),
                    width='stretch')


with tab3:
    st.caption("Live mini-sweep at reduced size (T=20, 10 inliers/sphere, "
               "seeds 0–1) around your current sidebar settings. Fast and "
               "indicative — report-grade numbers come from the notebook "
               "harness at full size.")

    VAR_GRIDS = {
        "n_background":     [0, 25, 50, 100, 150, 200],
        "obs_noise_std":    [0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
        "motion_noise_std": [0.0, 0.02, 0.05, 0.1, 0.2],
        "n_spheres":        [2, 3, 4, 6, 8],
        "speed":            [0.5, 1.0, 1.5, 2.0, 3.0],
    }
    var = st.selectbox("sweep variable", list(VAR_GRIDS))
    key = ("mini", var, params)

    if st.button("run mini sweep"):
        grid, seeds_mini = VAR_GRIDS[var], [0, 1]
        out, done = [], 0
        prog = st.progress(0.0)
        for v in grid:
            for s in seeds_mini:
                kw = dict(cube_size=L, n_spheres=K, radius=radius,
                          elasticity=1.0, speed=speed,
                          motion_noise_std=mnoise, n_timesteps=20, dt=dt,
                          n_inliers_per_sphere=10, n_background=25,
                          obs_noise_std=onoise, seed=s)
                kw[var] = v
                c = SimConfig(**kw)
                d = generate_dataset(c)
                cg_, cc_, mn_ = gate_key
                ins = build_instance(
                d, c_gate_spatial=cg_, c_gate_motion=cg_,
                c_cost_spatial=cc_, c_cost_motion=cc_,
                min_neighbors=(mn_ if mn_ > 0 else None),
                cyclic=False, verbose=False)

                t0 = time.perf_counter()
                lab, _ = greedy_solve(ins)
                rt_ = time.perf_counter() - t0
                inl = d.labels >= 0
                met_ = partition_metrics(d.labels[inl], lab[inl])
                out.append(dict(x=v, seed=s, runtime=rt_, **met_))
                done += 1
                prog.progress(done / (len(grid) * len(seeds_mini)),
                              text=f"{var} = {v}, seed {s}")
        prog.empty()
        st.session_state[key] = out

    res = st.session_state.get(key)
    if res is None:
        st.info("Pick a variable and press **run mini sweep** (~15–40 s).")
    else:
        xs_ = sorted({r_["x"] for r_ in res})

        def agg(metric):
            mu = np.array([np.mean([r_[metric] for r_ in res if r_["x"] == x])
                           for x in xs_])
            sd = np.array([np.std([r_[metric] for r_ in res if r_["x"] == x])
                           for x in xs_])
            return mu, sd

        vi_m, vi_s = agg("vi")
        rt_m, rt_s = agg("runtime")

        figs = go.Figure(go.Scatter(
            x=xs_, y=vi_m, mode="lines+markers", name="VI",
            error_y=dict(type="data", array=vi_s),
            line=dict(color="#00E5FF", width=3), marker=dict(size=9)))
        figs.update_layout(height=380, xaxis_title=var,
                           yaxis_title="VI (inliers, lower = better)",
                           title=f"GAEC accuracy vs {var}")
        st.plotly_chart(figs, width='stretch')

        figr = go.Figure(go.Scatter(
            x=xs_, y=rt_m, mode="lines+markers", name="runtime",
            error_y=dict(type="data", array=rt_s),
            line=dict(color="#FFD166", width=3), marker=dict(size=9)))
        figr.update_layout(height=320, xaxis_title=var,
                           yaxis_title="solve time [s]",
                           title=f"GAEC runtime vs {var}")
        st.plotly_chart(figr, width='stretch')
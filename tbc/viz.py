import numpy as np
import plotly.graph_objects as go
from tbc.motion import Trajectory
from tbc.synthesis import SimConfig, SyntheticDataset

COLORS = [
    "#E94F37", "#2E86AB", "#F4B942", "#44BBA4",
    "#9B5DE5", "#F15BB5", "#00BBF9", "#F77F00",
]


def _nan_break(arr: np.ndarray, L: float) -> np.ndarray:
    """Insert NaN rows where a trajectory wraps around the torus boundary,
    so Plotly doesn't draw a diagonal line across the box."""
    out = arr.astype(float).copy()
    jumps = np.where(np.any(np.abs(np.diff(arr, axis=0)) > L / 2, axis=1))[0]
    # Insert NaNs from the back so indices stay valid
    for j in sorted(jumps, reverse=True):
        nan_row = np.full((1, arr.shape[1]), np.nan)
        out = np.concatenate([out[:j+1], nan_row, out[j+1:]], axis=0)
    return out


def plot_spacetime(traj: Trajectory, cfg: SimConfig) -> go.Figure:
    """3D plot of (x, y, t) spacetime. Each sphere is a colored curve."""
    T = cfg.n_timesteps
    L = cfg.cube_size
    t_axis = np.arange(T) * cfg.dt
    traces = []

    for k in range(cfg.n_spheres):
        xy = traj.centers[:, k, :]          # (T, 2)
        broken = _nan_break(xy, L)          # (T+breaks, 2)
        t_vals = np.linspace(t_axis[0], t_axis[-1], len(broken))
        color = COLORS[k % len(COLORS)]
        traces.append(go.Scatter3d(
            x=broken[:, 0], y=broken[:, 1], z=t_vals,
            mode="lines+markers",
            name=f"Sphere {k}",
            line=dict(color=color, width=3),
            marker=dict(color=color, size=2),
        ))

    # Cube wireframe
    def edge(x0,y0,z0,x1,y1,z1):
        return [x0,x1,None], [y0,y1,None], [z0,z1,None]

    wx, wy, wz = [], [], []
    corners = [(0,0),(L,0),(L,L),(0,L),(0,0)]
    for z in [0, t_axis[-1]]:
        for i in range(4):
            ex,ey,ez = edge(corners[i][0],corners[i][1],z,
                            corners[i+1][0],corners[i+1][1],z)
            wx+=ex; wy+=ey; wz+=ez
    for cx,cy in [(0,0),(L,0),(L,L),(0,L)]:
        ex,ey,ez = edge(cx,cy,0,cx,cy,t_axis[-1])
        wx+=ex; wy+=ey; wz+=ez

    traces.append(go.Scatter3d(
        x=wx, y=wy, z=wz,
        mode="lines", name="boundary",
        line=dict(color="gray", width=1),
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Spacetime (x, y, t) — ground-truth trajectories",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="time",
            aspectmode="cube",
            xaxis=dict(range=[0, L]),
            yaxis=dict(range=[0, L]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_spacetime_dataset(ds: SyntheticDataset,
                           show_inliers: bool = True,
                           show_background: bool = True) -> go.Figure:
    """Same spacetime view but showing the noisy observed point cloud."""
    L   = ds.config.cube_size
    T   = ds.config.n_timesteps
    dt  = ds.config.dt
    traces = []

    if show_inliers:
        for k in range(ds.config.n_spheres):
            mask = ds.labels == k
            traces.append(go.Scatter3d(
                x=ds.points[mask, 0],
                y=ds.points[mask, 1],
                z=ds.times[mask] * dt,
                mode="markers",
                name=f"Sphere {k}",
                marker=dict(size=2, color=COLORS[k % len(COLORS)], opacity=0.6),
            ))

    if show_background:
        mask = ds.labels == -1
        traces.append(go.Scatter3d(
            x=ds.points[mask, 0],
            y=ds.points[mask, 1],
            z=ds.times[mask] * dt,
            mode="markers",
            name="background",
            marker=dict(size=1.5, color="lightgray", opacity=0.3),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Spacetime — observed point cloud (what the tracker sees)",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="time",
            aspectmode="cube",
            xaxis=dict(range=[0, L]),
            yaxis=dict(range=[0, L]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def animate_world(traj: Trajectory, cfg: SimConfig) -> go.Figure:
    """Animated 2D top-down view of sphere centers across frames."""
    T = cfg.n_timesteps
    L = cfg.cube_size

    frames = []
    margin = cfg.radius          # how far outside the box a ghost is still drawn
    for t in range(T):
        base = traj.centers[t]   # (K, 2)
        xs = list(base[:, 0])
        ys = list(base[:, 1])
        cs = [COLORS[k % len(COLORS)] for k in range(cfg.n_spheres)]

        # ghost copies: same sphere shifted by ±L, kept only near the box
        for k in range(cfg.n_spheres):
            for dx in (-L, 0, L):
                for dy in (-L, 0, L):
                    if dx == 0 and dy == 0:
                        continue
                    gx, gy = base[k, 0] + dx, base[k, 1] + dy
                    if -margin < gx < L + margin and -margin < gy < L + margin:
                        xs.append(gx)
                        ys.append(gy)
                        cs.append(COLORS[k % len(COLORS)])

        frames.append(go.Frame(
            data=[go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(
                    size=14,
                    color=cs,
                    line=dict(width=1, color="white"),
                ),
            )],
            name=str(t),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="2D world view (animated)",
            xaxis=dict(range=[0, L], title="x", constrain="domain"),
            yaxis=dict(range=[0, L], title="y", scaleanchor="x"),
            width=500, height=500,
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.1, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                          transition=dict(duration=0),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],

            )], sliders=[dict(
                steps=[dict(method="animate", args=[[str(t)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                            label=str(t)) for t in range(T)],
                x=0, y=0, len=1.0,
            )],),)
    return fig

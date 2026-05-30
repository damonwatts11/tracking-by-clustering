import numpy as np
from tbc.synthesis import SimConfig
from tbc.geometry import wrap
from tbc.motion import Trajectory


def sample_observations(traj: Trajectory, cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, times, labels) where:
    points : (M, 2) float -- 2D spatial coordinates
    times  : (M,)  int   -- frame index in {0, ..., T-1}
    labels : (M,)  int   -- k for sphere k, -1 for background
    and M = T * (K * n_inliers_per_sphere + n_background).
    """
    T   = cfg.n_timesteps
    K   = cfg.n_spheres
    nin = cfg.n_inliers_per_sphere
    nbg = cfg.n_background
    L   = cfg.cube_size
    sig = cfg.obs_noise_std

    M = T * (K * nin + nbg)
    points = np.empty((M, 2))
    times  = np.empty(M, dtype=int)
    labels = np.empty(M, dtype=int)

    idx = 0
    for t in range(T):
        # --- inliers: nin noisy points around each sphere center ---
        centers_t = traj.centers[t]           # (K, 2)
        noise = rng.normal(0, sig, size=(K, nin, 2))
        inlier_pts = wrap(centers_t[:, None, :] + noise, L)  # (K, nin, 2)
        inlier_pts = inlier_pts.reshape(K * nin, 2)

        n_in = K * nin
        points[idx: idx + n_in] = inlier_pts
        times [idx: idx + n_in] = t
        labels[idx: idx + n_in] = np.repeat(np.arange(K), nin)
        idx += n_in

        # --- background: nbg uniform random points ---
        bg_pts = rng.uniform(0, L, size=(nbg, 2))
        points[idx: idx + nbg] = bg_pts
        times [idx: idx + nbg] = t
        labels[idx: idx + nbg] = -1
        idx += nbg

    return points, times, labels
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SimConfig:
    cube_size:            float = 10.0
    n_spheres:            int   = 4
    radius:               float = 0.5
    elasticity:           float = 1.0
    speed:                float = 1.0
    motion_noise_std:     float = 0.05
    n_timesteps:          int   = 60
    dt:                   float = 0.1
    n_inliers_per_sphere: int   = 20
    n_background:         int   = 50
    obs_noise_std:        float = 0.1
    seed:                 int   = 0


@dataclass
class SyntheticDataset:
    points:     np.ndarray
    times:      np.ndarray
    labels:     np.ndarray
    trajectory: object     # Trajectory — imported lazily to avoid circular import
    config:     SimConfig


def generate_dataset(cfg: SimConfig) -> SyntheticDataset:
    from tbc.motion import simulate
    from tbc.observation import sample_observations
    rng    = np.random.default_rng(cfg.seed)
    traj   = simulate(cfg)
    points, times, labels = sample_observations(traj, cfg, rng)
    return SyntheticDataset(points=points, times=times, labels=labels,
                            trajectory=traj, config=cfg)


def save_dataset(ds: SyntheticDataset, path: str) -> None:
    np.savez_compressed(
        path,
        points=ds.points,
        times=ds.times,
        labels=ds.labels,
        centers=ds.trajectory.centers,
        velocities=ds.trajectory.velocities,
        **{f"cfg_{k}": v for k, v in asdict(ds.config).items()}
    )


def load_dataset(path: str) -> SyntheticDataset:
    from tbc.motion import Trajectory
    d    = np.load(path, allow_pickle=False)
    cfg  = SimConfig(**{k[4:]: d[k].item() for k in d if k.startswith("cfg_")})
    traj = Trajectory(centers=d["centers"], velocities=d["velocities"])
    return SyntheticDataset(points=d["points"], times=d["times"],
                            labels=d["labels"], trajectory=traj, config=cfg)
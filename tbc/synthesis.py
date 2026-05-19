from dataclasses import dataclass
@dataclass
class SimConfig:
    cube_size: float = 10.0
    n_spheres: int = 4
    radius: float = 0.5
    elasticity: float = 1.0
    speed: float = 1.0
    motion_noise_std: float = 0.05
    n_timesteps: int = 60
    dt: float = 0.1
    n_inliers_per_sphere: int = 20
    n_background: int = 50
    obs_noise_std: float = 0.1
    seed: int = 0
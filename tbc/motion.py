import numpy as np
from dataclasses import dataclass
from tbc.synthesis import SimConfig
from tbc.collision import resolve_collisions
from tbc.geometry import torus_distance, wrap
def sample_initial_centers(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Return centers of shape (K, 2) such that no two are closer than 2*radius on the 2-torus. Use rejection sampling."""
    L = cfg.cube_size
    k = cfg.n_spheres
    r = cfg.radius
    max_attempts = 1000 * k
    centers = []
    attempts = 0
    while len(centers) < k and attempts < max_attempts:
        candidate = rng.uniform(0, L, size=2)
        if len(centers) == 0 or torus_distance(candidate, np.array(centers), L).min() > 2 * r:
            centers.append(candidate)
        attempts += 1

    if len(centers) < k:
        raise RuntimeError(
            f"Could not put {k} non-overlapping spheres after {max_attempts} attempts. "
            f"Try fewer spheres, a smaller radius, or a larger cube_size."
        )
    return np.array(centers)   
            
def sample_initial_velocities(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Return velocities of shape (K, 2): random direction, magnitude = cfg.speed."""
    k = cfg.n_spheres
    g = rng.normal(size=(k, 2))                             
    norms = np.linalg.norm(g, axis=1, keepdims=True)        
    directions = g / norms                                   
    return directions * cfg.speed                            

def step_motion ( centers : np . ndarray , velocities : np . ndarray , cfg : SimConfig , rng : np . random . Generator ) -> tuple [ np .ndarray , np . ndarray ]:
    """ One Euler step : 
           1. velocities += Gaussian noise with std cfg . motion_noise_std
           2. centers += velocities * cfg .dt
           3. wrap centers into [0 , cube_size )
        Return ( new_centers , new_velocities ). Both arrays of shape (K,2).
    """
    new_velocities = velocities + rng.normal(0, cfg.motion_noise_std, size=velocities.shape)
    new_centers = wrap(centers + new_velocities * cfg.dt, cfg.cube_size)
    return new_centers, new_velocities

@dataclass
class Trajectory:
    centers: np.ndarray     # shape (T, K, 2) where T is the time steps, K is the number of spheres and 2 for the spatial coordinates (x,y)
    velocities: np.ndarray  # shape (T, K, 2)


def simulate(cfg: SimConfig) -> Trajectory:
    """Run the full simulation. Returns the ground-truth trajectory.
    The time axis is treated as cyclic with period T by the downstream tracker (since 3 torus 2 dimensions + time axis),
    but the simulation itself runs forward in time with no periodicity
    constraint on trajectories."""

    rng = np.random.default_rng(cfg.seed)
    T = cfg.n_timesteps
    K = cfg.n_spheres
    centers_hist    = np.empty((T, K, 2))
    velocities_hist = np.empty((T, K, 2))

    centers    = sample_initial_centers(cfg, rng)
    velocities = sample_initial_velocities(cfg, rng)
    centers_hist[0]    = centers
    velocities_hist[0] = velocities

    for t in range(1, T):
        centers, velocities = step_motion(centers, velocities, cfg, rng)
        velocities          = resolve_collisions(centers, velocities, cfg)
        centers_hist[t]    = centers
        velocities_hist[t] = velocities

    return Trajectory(centers=centers_hist, velocities=velocities_hist)
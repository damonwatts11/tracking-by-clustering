from itertools import combinations 
import numpy as np
from tbc.synthesis import SimConfig
from tbc.geometry import min_image_displacement

def resolve_collisions(centers: np.ndarray, velocities: np.ndarray,
                       cfg: SimConfig) -> np.ndarray:
    """Return updated velocities after resolving all pairwise overlaps."""
    L   = cfg.cube_size
    r   = cfg.radius
    K   = cfg.n_spheres
    e = cfg.elasticity
    velocity_deltas = np.zeros_like(velocities)
    
    for i, j in combinations(range(K), 2):
        
        delta = min_image_displacement(centers[i], centers[j], L)
        dist  = np.linalg.norm(delta)
        if dist >= 2 * r:
            continue

        normal_vect = delta / dist
        rel_normal = np.dot(velocities[i] - velocities[j], normal_vect)
        if rel_normal <= 0:
            continue                                         
        
        dv = e * rel_normal * normal_vect
        velocity_deltas[i] -= dv
        velocity_deltas[j] += dv
    return velocities + velocity_deltas

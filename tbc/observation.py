import numpy as np
from tbc.synthesis import SimConfig
from tbc.geometry import wrap, min_image_displacement, torus_distance
from tbc.motion import (
    sample_initial_centers,
    sample_initial_velocities,
    step_motion,
    simulate,
    Trajectory,
)
from tbc.collision import resolve_collisions

def sample_observations ( traj : Trajectory , cfg : SimConfig , rng : np . random . Generator ) -> tuple [ np . ndarray , np . ndarray , np . ndarray ]:
    """ Return (points , times , labels ) where :
    points : (M, 2) float -- 2D spatial coordinates
    times : (M ,) int -- frame index in {0 , ... , T -1}
    labels : (M ,) int -- k for sphere k, -1 for background
    and M = T * (K * n_inliers_per_sphere + n_background ).
    """
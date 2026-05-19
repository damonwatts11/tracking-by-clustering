import numpy as np

def wrap ( x : np . ndarray , L : float ) -> np . ndarray :
    """ Wrap coordinates of any shape into [0 , L)."""
    return np.mod(x,L)
  

def min_image_displacement ( a : np . ndarray , b : np . ndarray , L : float ) -> np . ndarray :
    """ Return the shortest displacement from a to b under torus topology .
    Works for arrays a, b with matching shapes ending in 2. """
    return np.mod(b-a+L/2,L)-L/2


def torus_distance ( a : np . ndarray , b : np . ndarray , L : float ) -> np . ndarray :
    """ Euclidean length of the minimum - image displacement ."""
    return np.linalg.norm(min_image_displacement(a,b,L), axis=-1)



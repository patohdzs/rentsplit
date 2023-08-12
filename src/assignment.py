from itertools import permutations

import numpy as np
from properties import is_efficient


def compute_efficient_assignments(V: np.ndarray) -> list[tuple[int]]:
    """Computes set of Pareto-efficient room assignments.

    Args:
        V (np.ndarray): Matrix of room valuations.

    Returns:
        list[tuple[int]]: List of PE assignments.
    """
    assignments = permutations(range(V.shape[0]))
    return [mu for mu in assignments if is_efficient(mu, V)]

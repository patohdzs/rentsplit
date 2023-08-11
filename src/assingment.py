from itertools import permutations

import numpy as np
from allocation import is_efficient


def compute_efficient_assingments(V: np.ndarray):
    assingments = permutations(range(V.shape[0]))
    return [mu for mu in assingments if is_efficient(mu, V)]

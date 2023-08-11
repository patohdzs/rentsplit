import numpy as np
from scipy.optimize import linear_sum_assignment


def is_envy_free(mu: tuple[int], p: np.ndarray, V: np.ndarray, verbose: bool = False):
    for i, m in enumerate(mu):
        for j, n in enumerate(mu):
            ef_cond = V[i][m] - p[m] >= V[i][n] - p[n]
            if not ef_cond:
                if verbose:
                    print("Wait...")
                    print(f"Agent {i+1} has room {m +1}, but prefers room {n +1}")
                    print("Utility from m", V[i][m] - p[m])
                    print("Utility from n", V[i][n] - p[n])
                return False
    return True


def is_efficient(mu: tuple[int], V: np.ndarray) -> bool:
    n = V.shape[0]
    i, j = linear_sum_assignment(V, maximize=True)
    social_surplus = V[i, j].sum()

    return V[range(n), mu].sum() == social_surplus

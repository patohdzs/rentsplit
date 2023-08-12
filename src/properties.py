import numpy as np
from scipy.optimize import linear_sum_assignment


def is_envy_free(
    mu: tuple[int], p: np.ndarray, V: np.ndarray, verbose: bool = False
) -> bool:
    """Checks if an allocation is envy-free.

    Args:
        mu (tuple[int]): Allocation room assignment.
        p (np.ndarray): Allocation room prices.
        V (np.ndarray): Matrix of room valuations.
        verbose (bool, optional): Verbosity. Defaults to False.

    Returns:
        bool: True for EF allocations.
    """
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
    """Checks if a room assignment is Pareto-efficient.

    Args:
        mu (tuple[int]): Room assignment to check.
        V (np.ndarray): Matrix of room valuations.

    Returns:
        bool: True for PE assignments.
    """
    n = V.shape[0]
    i, j = linear_sum_assignment(V, maximize=True)
    social_surplus = V[i, j].sum()

    return V[range(n), mu].sum() == social_surplus

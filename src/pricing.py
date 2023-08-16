import numpy as np


def compute_auction_prices(V: np.ndarray, R: int, tol: int = 10**5) -> np.ndarray:
    """Computes envy-free prices using a simultanous
    ascending-descending auctions mechanism,
    following  Abdulkadiroğlu et al. (2004).

    Args:
        V (np.ndarray): Matrix of room valuations.
        R (int): Total rent.
        tol (int, optional): Maximum number of iterations. Defaults to 10**5.

    Raises:
        Exception: _description_

    Returns:
        np.ndarray: Vector of envy-free room prices.
    """
    # Initialize variables
    n = V.shape[0]
    delta = 1 / (8 * n)
    p = (R / n) * np.ones(n)
    t = 0
    while t < tol:
        # Find over-demanded rooms
        od = _find_overdemanded(V, p)
        if (od.size == 0) or (od.size == n):
            # Re-normalize
            p = p + (R - p.sum()) / n
            return p

        # Increase prices of over demanded
        p[od] += delta

        # Decrease prices of not-over-demanded
        not_od = np.setdiff1d(range(n), od)
        p[not_od] -= (len(od) / (n - len(od))) * delta

        # Increase counter
        t += 1
    raise Exception(f"TIMEOUT after {tol} iterations.")


def _find_overdemanded(V: np.ndarray, p: np.ndarray):
    # Get demands as numpy array
    D = _demands(V, p)
    carryon = True

    while carryon:
        mask = _iteration(D)
        if mask is not None:
            D = D[~mask, :]
        else:
            carryon = False

    return np.unique(D[:, 1])


def _iteration(D: np.ndarray) -> np.ndarray | None:
    # Get unique and counts
    unique, counts = np.unique(D[:, 1], return_counts=True)

    # Get rooms with count == 1
    underdemanded = unique[counts == 1]

    # If no rooms are underdemanded, you're done
    if underdemanded.size == 0:
        return None

    # Remove agents that demand underdemanded rooms
    agents_with_underdemanded = D[np.isin(D[:, 1], underdemanded), 0]
    return np.isin(D[:, 0], agents_with_underdemanded)


def _demands(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Find utilities
    U = V - p
    max_u = np.max(U, axis=1)

    # Find demanded
    demanded = (U.T == max_u).T & (U >= 0)

    # Get demands as numpy array
    return np.transpose((demanded).nonzero())

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
        if not od:
            # Re-normalize
            p = p + (R - p.sum()) / n
            return p

        # Increase prices of over demanded
        p[od] += delta

        # Decrease prices of not-over-demanded
        not_od = np.setdiff1d(range(n), od)
        p[not_od] -= (n - len(od)) / len(od) * delta

        # Increase counter
        t += 1
    raise Exception(f"TIMEOUT after {tol} iterations.")


def _find_overdemanded(V: np.ndarray, p: np.ndarray) -> list[int]:
    # Find demands
    utilities = V - p
    max_utility = np.max(utilities, axis=1)
    demands = [
        room
        for i, row in enumerate(utilities)
        for room, u in enumerate(row)
        if u == max_utility[i] and u >= 0
    ]
    # Return over-demanded rooms
    return [x for x in set(demands) if demands.count(x) > 1]

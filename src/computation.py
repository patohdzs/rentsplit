import numpy as np


def compute_auction_prices(V: np.ndarray, R: int, tol=1e5):
    # Initialize variables
    n = V.shape[0]
    delta = 1 / (8 * n)
    p = (R / n) * np.ones(n)
    t = 0
    while t < tol:
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
        t += 1
    raise Exception(f"TIMEOUT after {tol} iterations.")


def _find_overdemanded(V, p):
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

import numpy as np


def compute_auction_prices(
    V: np.ndarray, R: float, max_iter: int = 10**4
) -> np.ndarray:
    """Computes envy-free prices using a simultanous
    ascending-descending auctions mechanism,
    following  Abdulkadiroğlu et al. (2004).

    Args:
        V (np.ndarray): Matrix of room valuations.
        R (int): Total rent.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10**4.

    Returns:
        np.ndarray: Vector of envy-free room prices.
    """
    # Input validation
    V = np.asarray_chkfinite(V)

    if not isinstance(R, (int, float)):
        raise TypeError(f"Expected numeric (float, int), but R is {type(R)}")

    if len(V.shape) != 2:
        raise ValueError(
            "Expected a two-dimensional array (matrix)"
            + f", but the shape of V is {V.shape}"
        )

    if not V.shape[0] == V.shape[1]:
        raise ValueError(
            "Expected a square array (matrix)"
            + f", but the dimensions of V are {V.shape}"
        )

    if not (V.sum(axis=1) >= R).all():
        raise ValueError(
            "Expected a right-stochastic array (matrix)"
            + f", but the row-wise sums of V are {V.sum(axis=1)}"
        )

    if R <= 0:
        raise ValueError(f"Expected a positive scalar, but R is {R}")

    # Initialize variables
    n = V.shape[0]
    p = (R / n) * np.ones(n)
    t = 0
    while t < max_iter:
        # Find over-demanded rooms
        od = _get_overdemanded(V, p)
        if (od.size == 0) or (od.size == n):
            # Re-normalize
            p = p + (R - p.sum()) / n
            return p

        # Find increment
        x = _get_increment(V, p)

        # Increase prices of over demanded
        p[od] += ((n - len(od)) / n) * x

        # Decrease prices of not-over-demanded
        not_od = np.setdiff1d(range(n), od)
        p[not_od] -= (len(od) / n) * x

        # Increase counter
        t += 1
    raise RuntimeError("Maximum number of iterations reached.")


def _get_increment(V, p):
    # Utility
    U = V - p

    # Utility tilde (max utility)
    max_u = np.max(U, axis=1)

    # Overdemanded and not overdemanded rooms
    od = _get_overdemanded(V, p)
    not_od = np.setdiff1d(range(V.shape[0]), od)

    # Find J
    D = _get_demands(V, p)
    J = [i for i in range(V.shape[0]) if np.isin(D[D[:, 0] == i, 1], od).all()]

    # Inner part of the problem
    inner = max_u - np.take(U, not_od, axis=1).max(axis=1)

    # Return x
    return inner[J].min()


def _get_overdemanded(V: np.ndarray, p: np.ndarray):
    # Get demands as numpy array
    D = _get_demands(V, p)
    carry_on = True

    # Remove underdemanded rooms iteratively
    while carry_on:
        mask = _is_underdemanded(D)
        if mask is not None:
            D = D[~mask, :]
        else:
            carry_on = False

    return np.unique(D[:, 1])


def _is_underdemanded(D: np.ndarray) -> np.ndarray | None:
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


def _get_demands(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Find utilities
    U = V - p
    max_u = np.max(U, axis=1)

    # Find demanded
    demanded = (U.T == max_u).T & (U >= 0)

    # Get demands as numpy array
    return np.transpose((demanded).nonzero())

import numpy as np
import pytest
from assignment import compute_efficient_assignments
from pricing import compute_auction_prices
from properties import is_envy_free

# TODO: test envy-freeness 2x2 case and 3x3 case with simulated matrices
# TODO: test envy free-ness


@pytest.mark.parametrize(
    "V, R",
    [
        (np.array([[700, 300], [600, 400]]), 1000),
        (np.array([[400, 400, 200], [500, 400, 100], [700, 200, 100]]), 1000),
        (
            np.array(
                [
                    [900, 890, 900, 810],
                    [900, 890, 890, 820],
                    [904, 896, 904, 796],
                    [900, 875, 875, 850],
                ]
            ),
            3500,
        ),
        (
            np.array(
                [
                    [890, 900, 900, 810],
                    [885, 890, 890, 835],
                    [905, 892, 894, 809],
                    [885, 895, 870, 850],
                ]
            ),
            3500,
        ),
        (
            np.array(
                [
                    [15, 18, 10, 15, 24, 28],
                    [18, 25, 3, 18, 25, 15],
                    [6, 25, 15, 18, 18, 25],
                    [18, 5, 18, 12, 9, 25],
                    [6, 22, 5, 5, 10, 12],
                    [6, 9, 2, 21, 25, 9],
                ]
            ),
            60,
        ),
    ],
)
def test_examples_ef(V: np.ndarray, R: float):
    # Compute assingments
    mus = compute_efficient_assignments(V)

    # Compute prices
    p = compute_auction_prices(V, R)
    for mu in mus:
        assert is_envy_free(mu, p, V) is True

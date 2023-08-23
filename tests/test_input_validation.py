import numpy as np
import pytest
from pricing import compute_auction_prices


@pytest.mark.parametrize("R", ["test1", "12"])
def test_validation_rent(R: float):
    V = np.array([[700, 300], [600, 400]])
    with pytest.raises(TypeError):
        compute_auction_prices(V, R)

from dataclasses import dataclass

import numpy as np


@dataclass
class RentDivisionProblem:
    V: np.ndarray
    R: int

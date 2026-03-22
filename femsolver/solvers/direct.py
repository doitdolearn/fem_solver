from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class DirectSolver:
    """A wrapper of spsolve"""
    def solve(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        return spsolve(K, F)
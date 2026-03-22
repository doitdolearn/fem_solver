from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def compute_reactions(K: csr_matrix, u_full: np.ndarray, F_ext: np.ndarray) -> np.ndarray:
    """ R = KU - F_ext"""
    return np.asarray(K @ u_full).flatten() - F_ext
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def partition_system(
    K: csr_matrix,
    F: np.ndarray,
    free_dofs: List[int],
    constrained_dofs: List[int],
    constrained_values: Dict[int, float],
    solve_fn: Optional[Callable] = None,
) -> np.ndarray:
    """
    Apply Essential BCs via the partition method and return the displacement vector

    Solve: K_ff * u_f = F_f - K_fc * u_c
    """
    n = K.shape[0]
    u_c = np.array([constrained_values.get(d, 0.0) for d in constrained_dofs])

    if not free_dofs:
        u_full = np.zeros(n)
        u_full[constrained_dofs] = u_c
        return u_full

    K_ff = K[free_dofs, :][:, free_dofs]
    K_fc = K[free_dofs, :][:, constrained_dofs]
    F_f = F[free_dofs]

    rhs = F_f - np.asarray(K_fc @ u_c).flatten()

    _solve = solve_fn if solve_fn is not None else spsolve
    u_f = _solve(K_ff, rhs)

    u_full = np.zeros(n)
    u_full[free_dofs] = np.asarray(u_f).flatten()
    u_full[constrained_dofs] = u_c
    return u_full
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class NewtonRaphsonSolver:
    """Newton-Raphson iteration loop for nonlinear static problems.

    Solve the nonlinear residual equation "R(u_f) = F_int(u_f) - F_ext = 0"
    for the free-DOF vector u_f.

    Convergence criterion:  ‖R_f‖ / max(‖F_ext_f‖, ε_abs) < tol

    Parameters
    ----------
    tol:      Relative convergence tolerance (default 1e-10).
    max_iter: Maximum number of NR iterations per load step (default 50).
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 50):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self,
              compute_KT_and_R: Callable[[np.ndarray], Tuple[csr_matrix, np.ndarray]],
              u_f_init: np.ndarray,
              F_ext_f: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """Run Newton-Raphson iteration until convergence or max_iter.

        :param compute_KT_and_R: Callable
            Given the current free-DOF displacement u_f, returns (K_T_ff sparse matrix, R_f residual vector)
        :param u_f_init: ndarray
            Starting guess for the free-DOF displacement vector
        :param F_ext_f: ndarray
            External force on free DOFs (Used as normalization for residual)
        :return:
        u_f: ndarray
            converged( or best effort) free-DOF displacement
        n_iters: int
            number of iterations performed
        converged: True if converged, False otherwise
        """
        u_f = u_f_init.copy()
        F_norm = max(float(np.linalg.norm(F_ext_f)), 1e-14)

        for n_iter in range(self.max_iter):
            K_T_ff, R_f = compute_KT_and_R(u_f)

            if np.linalg.norm(R_f) / F_norm < self.tol:
                return u_f, n_iter, True

            du_f = spsolve(K_T_ff, -R_f)
            u_f = u_f + np.asarray(du_f).flatten()
        return u_f, n_iter, False
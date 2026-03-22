from __future__ import annotations

from typing import List, Tuple

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_2d

class Quad4AxisymElement(BaseElement):
    """4-node axisymmetric quadrilateral element in (r, z) coordinates.
     ** for natural coordinates, (q, s) are used instead of (r,s) in other elements to avoid confusing

    Bilinear shape functions with 2x2 Gauss quadrature.
    2 DOFs per node: (ur, uz) -> 8 DOFs total.

    Strain components: [epsilon_rr, epsilon_tt, epsilon_zz, gamma_rz]
      epsilon_rr = du_r/dr
      epsilon_tt = u_r/r        (hoop strain from axisymmetry)
      epsilon_zz = du_z/dz
      gamma_rz = du_r/dz + du_z/dr

    Volume element: dV = 2*pi*r * dr * dz  (per-radian: r * det(J) * dq * ds)
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, delta_T: float = 0.0, **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        self.delta_T = delta_T
        self._gp_pts, self._gp_wts = gauss_2d(2)

    # Shape functions and derivatives
    def shape_functions(self, q: float, s: float) -> np.ndarray:
        """Bilinear shape functions N (4,) at (q, s)."""
        return 0.25 * np.array([
            (1.0 - q) * (1.0 - s),
            (1.0 + q) * (1.0 - s),
            (1.0 + q) * (1.0 + s),
            (1.0 - q) * (1.0 + s),
        ])

    def _dN_dq_ds(self, q: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Derivatives dN/dq (4,), dN/ds (4,)."""
        dNdq = 0.25 * np.array([
            -(1.0 - s), (1.0 - s), (1.0 + s), -(1.0 + s)
        ])
        dNds = 0.25 * np.array([
            -(1.0 - q), -(1.0 + q), (1.0 + q), (1.0 - q)
        ])
        return dNdq, dNds

    def _jacobian(self, q: float, s: float) -> np.ndarray:
        """2x2 jacobian: (q, s) -> (r, z)"""
        dNdq, dNds = self._dN_dq_ds(q, s)
        r = self.nodes_coords[:, 0]
        z = self.nodes_coords[:, 1]
        return np.array([
            [dNdq @ r, dNdq @ z],
            [dNds @ r, dNds @ z]
        ])

    # B matrix (axisymmetric)
    def _B_and_detJ_r(self, q: float, s: float):
        N = self.shape_functions(q, s)
        dNdq, dNds = self._dN_dq_ds(q, s)
        J = self._jacobian(q, s)
        det_J = float(np.linalg.det(J))
        J_inv = np.linalg.inv(J)

        dN_drz = J_inv @ np.vstack([dNdq, dNds])
        dNdr = dN_drz[0]
        dNdz = dN_drz[1]

        r = float(N @ self.nodes_coords[:, 0])

        B = np.zeros((4, 8))
        inv_r = 1.0/r
        for i in range(4):
            c = 2 * i
            # epsilon_rr = du_r/dr
            B[0, c] = dNdr[i]
            # epsilon_tt = u_r/r
            B[1, c] = N[i] * inv_r
            # epsilon_zz = du_z/dz
            B[2, c + 1] = dNdz[i]
            # gamma_rz = du_r/dz + du_z/dr
            B[3, c] = dNdz[i]
            B[3, c + 1] = dNdr[i]
        return B, det_J, r

    def compute_B_matrix(self, q: float = 0.0, s: float = 0.0) -> np.ndarray:
        B, _, _ = self._B_and_detJ_r(q, s)
        return B

    # Element matrices
    def compute_k_e(self) -> np.ndarray:
        C = self.material.constitutive_matrix_axisym()
        K = np.zeros((8, 8))
        for (q, s), w in zip(self._gp_pts, self._gp_wts):
            B, det_J, r = self._B_and_detJ_r(q, s)
            K += w * r * det_J * (B.T @ C @ B)
        return K

    def compute_f_e(self) -> np.ndarray:
        C = self.material.constitutive_matrix_axisym()
        alpha = self.material.alpha
        dT = self.delta_T
        # Thermal strain: radial, hoop, axial expansion; no shear
        eps_th = alpha * dT * np.array([1.0, 1.0, 1.0, 0.0])
        sigma_th = C @ eps_th

        f = np.zeros(8)
        for (q, s), w in zip(self._gp_pts, self._gp_wts):
            B, det_J, r = self._B_and_detJ_r(q, s)
            f += w * r * det_J * (B.T @ sigma_th)
        return f

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        """Stress at centroid: sigma = C * (B * u_e - eps_th)."""
        C = self.material.constitutive_matrix_axisym()
        B, _, _ = self._B_and_detJ_r(0.0, 0.0)
        eps_th = self.material.alpha * self.delta_T * np.array([1.0, 1.0, 1.0, 0.0])
        return C @ (B @ u_e - eps_th)
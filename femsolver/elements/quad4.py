from __future__ import annotations

from typing import Tuple

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_2d

class Quad4Element(BaseElement):
    """4 nodes bilinear isoparametric quadrilateral for 2D plane problems.

    DOF Order: [ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3]  (8 DOFs)

    Nodes ordered in natural coordinates (r, s) \in [-1, 1]^2
        Node 0: (-1, -1) -> BL
        Node 1: (+1, -1) -> BR
        Node 2: (+1, +1) -> TR
        Node 3: (-1, +1) -> TL

    Use 2*2 Gaussian Integration
    """

    def __init__(self, element_id, node_ids, nodes_coords, material,
                 plane: str = "stress", **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        self.plane = plane
        self.t = getattr(material, "thickness", 1.0) or 1.0

    # H matrix and Jacobian
    def shape_functions(self, r: float, s: float):
        """Return N (4, ) at (ri, si)"""
        return 0.25 * np.array(
            [
                (1.0 - r) * (1.0 - s),
                (1.0 + r) * (1.0 - s),
                (1.0 + r) * (1.0 + s),
                (1.0 - r) * (1.0 + s),
            ]
        )

    def _dN_dr_ds(self, r: float, s: float):
        dNdr = 0.25 * np.array(
            [-(1.0 - s), (1.0 - s), (1.0 + s), -(1.0 + s)]
        )
        dNds = 0.25 * np.array(
            [-(1.0 - r), -(1.0 + r), (1.0 + r), (1.0 - r)]
        )
        return dNdr, dNds

    def _jacobian(self, r: float, s: float):
        dNdr, dNds = self._dN_dr_ds(r, s)
        x = self.nodes_coords[:, 0]
        y = self.nodes_coords[:, 1]
        return np.array(
            [[dNdr @ x, dNdr @ y], [dNds @ x, dNds @ y]]
        )

    # B matrix
    def _B_and_det_J(self, r: float, s: float) -> Tuple[np.ndarray, float]:
        dNdr, dNds = self._dN_dr_ds(r, s)
        J = self._jacobian(r, s)
        det_J = float(np.linalg.det(J))
        if det_J <= 0:
            raise ValueError(f"Element {self.element_id} has non-positive Jacobian. \n"
                             f"At: r = {r: .3f}, s = {s: .3f}, det(J) = {det_J:.4e}. \n"
                             "Check the node ordering - nodes must be ordered in counter clockwise direction\n")
        J_inv = np.linalg.inv(J)
        # dN/dx, dN/dy via chain rule: [dN/dx; dN/dy] = J_inv * [dN/dr ; dN/ds]
        dN_dxy = J_inv @ np.vstack([dNdr, dNds])
        dNdx = dN_dxy[0]
        dNdy = dN_dxy[1]

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dNdx[i]  # \epsilon_xx
            B[1, 2 * i + 1] = dNdy[i]  # \epsilon_yy
            B[2, 2 * i] = dNdy[i]  # \gamma_xy
            B[2, 2 * i + 1] = dNdx[i]  # \gamma_xy
        return B, det_J

    def compute_B_matrix(self, r: float = 0.0, s: float = 0.0) -> np.ndarray:
        B, _ = self._B_and_det_J(r, s)
        return B

    def compute_k_e(self) -> np.ndarray:
        C = self.material.constitutive_matrix(self.plane)
        K = np.zeros((8, 8))

        points, weights = gauss_2d(2)
        for (r, s), w in zip(points, weights):
            B, det_J = self._B_and_det_J(r, s)
            K += w * self.t * det_J * (B.T @ C @ B)
        return K

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(8)

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        """Stress at the centroid (r=s=0): \sigma = C*B*u_e."""
        C = self.material.constitutive_matrix(self.plane)
        B, _ = self._B_and_det_J(0.0, 0.0)
        return C @ B @ u_e










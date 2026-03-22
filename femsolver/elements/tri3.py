from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement

class Tri3Element(BaseElement):
    """3 nodes constant strain triangle(CST) for 2d plane problems

    DOF order: [ux1, uy1, ux2, uy2, ux3, uy3]  (6 DOFs)
    Node order: counter-clockwise gives positive area.
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, plane: str = "stress", **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        self.plane = plane
        self.t = getattr(material, "thickness", 1.0) or 1.0

        x = nodes_coords[:, 0]
        y = nodes_coords[:, 1]
        # Signed area (positive for CCW node ordering)
        self._area_signed = 0.5 * ((x[1] - x[0]) * (y[2] - y[0])
                                   - (x[2] - x[0]) * (y[1] - y[0]))
        self.area = abs(self._area_signed)
        if self.area < 1e-14:
            raise ValueError(
                f"Element {element_id}: zero-area triangle at nodes {nodes_coords}"
            )

    def compute_B_matrix(self, *args) -> np.ndarray:
        x = self.nodes_coords[:, 0]
        y = self.nodes_coords[:, 1]
        A2 = 2.0 * self._area_signed

        y23 = y[1] - y[2]
        y31 = y[2] - y[0]
        y12 = y[0] - y[1]
        x32 = x[2] - x[1]
        x13 = x[0] - x[2]
        x21 = x[1] - x[0]

        return (1.0 / A2) * np.array(
            [
                [y23, 0, y31, 0, y12, 0],
                [0, x32, 0, x13, 0, x21],
                [x32, y23, x13, y31, x21, y12],
            ]
        )

    def compute_k_e(self) -> np.ndarray:
        C = self.material.constitutive_matrix(self.plane)
        B = self.compute_B_matrix()
        return self.t * self.area * (B.T @ C @ B)

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(6)

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        """σ = C * B * u_e [\sigma_xx, \sigma_yy, \sigma_xy]"""
        C = self.material.constitutive_matrix(self.plane)
        B = self.compute_B_matrix()
        return C @ B @ u_e



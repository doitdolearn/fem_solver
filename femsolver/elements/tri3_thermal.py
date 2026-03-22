from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement


class Tri3ThermalElement(BaseElement):
    """2D CST (constant-strain triangle) thermal conduction element.

    Node order: counter-clockwise gives positive area.
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        self.k = material.k_cond
        self.t = getattr(material, "thickness", 1.0) or 1.0
        self.rho = material.rho
        self.cp = material.cp

        x = nodes_coords[:, 0]
        y = nodes_coords[:, 1]
        # Signed area (positive for CCW node ordering)
        self._area_signed = 0.5 * (
            (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        )
        self.area = abs(self._area_signed)
        if self.area < 1e-14:
            raise ValueError(
                f"Element {element_id}: zero-area triangle at nodes {nodes_coords}"
            )

    def compute_B_matrix(self) -> np.ndarray:
        """Return the 2×3 temperature-gradient matrix B_th."""
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
            [[y23, y31, y12],
             [x32, x13, x21]]
        )

    def compute_k_e(self) -> np.ndarray:
        B = self.compute_B_matrix()
        return self.k * self.t * self.area * (B.T @ B)

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(3)

    def compute_stress(self, T_e: np.ndarray) -> np.ndarray:
        """Heat flux [qx, qy] = -k · B_th @ T_e   [W/m^2]."""
        B = self.compute_B_matrix()
        return -self.k * (B @ T_e)
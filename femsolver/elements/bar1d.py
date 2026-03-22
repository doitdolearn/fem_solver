from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_1d

class Bar1DElement(BaseElement):
    """2-node 1D bar element"""

    def __init__(self, element_id, node_ids, nodes_coords, material):
        super().__init__(element_id, node_ids, nodes_coords, material)
        x1 = float(self.nodes_coords[0, 0])
        x2 = float(self.nodes_coords[1, 0])
        self.L = abs(x2 - x1)
        if self.L < 1e-12:
            raise ValueError(f"Element {element_id} is a zero length")

        self.B = self.compute_B_matrix()
        self.E = self.material.E
        self.A = self.material.A
        self.J = self.L / 2.0 # Jacobian

    def compute_B_matrix(self) -> np.ndarray:
        return np.array([[-1.0 / self.L, 1.0 / self.L]])

    def shape_functions(self, r: float) -> np.ndarray:
        """ N1 = (1 - r)/2, N2 = (1 + r)/2 for r \in [-1, 1]"""
        return np.array([(1.0 - r) / 2.0, (1.0 + r) / 2.0])

    def compute_k_e(self) -> np.ndarray:
        """Implement via gaussian integration"""
        points, weights = gauss_1d(1)

        K = np.zeros((2, 2))
        for r, w in zip(points, weights):
            K += (self.B.T @ np.array([[self.E]]) @ self.B) * self.A * self.J * w
        return K

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(2)

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        """ \sigma = C * B * u_e """
        B = self.compute_B_matrix()
        E = self.material.E
        return np.array([[E]]) @ B @ u_e



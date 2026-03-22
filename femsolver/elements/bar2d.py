from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_1d


class Bar2DElement(BaseElement):
    """2-node 2D bar/truss element using Gauss-Legendre Quadrature.

    DOF ordering: [ux1, uy1, ux2, uy2]  (4 DOFs total)
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        dx = nodes_coords[1, 0] - nodes_coords[0, 0]
        dy = nodes_coords[1, 1] - nodes_coords[0, 1]
        self.L = np.sqrt(dx ** 2 + dy ** 2)
        if self.L < 1e-14:
            raise ValueError(
                f"Element {element_id}: zero-length element "
                f"(nodes at {nodes_coords[0]} and {nodes_coords[1]})"
            )
        self.c = dx / self.L  # cosine theta
        self.s = dy / self.L  # sine theta

        self.E = material.E
        self.A = material.A
        self.rho_A = material.rho * material.A
        self.J = self.L / 2.0  # Jacobian

    def shape_functions(self, r: float) -> np.ndarray:
        N1 = (1.0 - r) / 2.0
        N2 = (1.0 + r) / 2.0
        return np.array([
            [N1, 0.0, N2, 0.0],
            [0.0, N1, 0.0, N2]
        ])

    def compute_B_matrix(self, r: float = 0.0) -> np.ndarray:
        c, s = self.c, self.s
        return (1.0 / self.L) * np.array([[-c, -s, c, s]])

    def compute_k_e(self) -> np.ndarray:
        points, weights = gauss_1d(1)

        K = np.zeros((4, 4))
        for r, w in zip(points, weights):
            B = self.compute_B_matrix(r)
            K += (B.T @ np.array([[self.E]]) @ B) * self.A * self.J * w

        return K

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(4)

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        """\sigma = E * B * u_e"""
        B = self.compute_B_matrix()
        E = self.material.E
        return np.array([[E]]) @ B @ u_e
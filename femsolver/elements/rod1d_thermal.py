from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_1d

class Rod1DThermalElement(BaseElement):
    """1D rod thermal conduction element using Gaussian integration

    DOF order: [T1, T2]
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        x1 = float(self.nodes_coords[0, 0])
        x2 = float(self.nodes_coords[1, 0])
        self.L = abs(x2 - x1)
        if self.L < 1e-14:
            raise ValueError(
                f"Element {element_id}: zero-length element "
                f"(node coords {x1} and {x2} are identical)"
            )

        self.k = material.k_cond
        self.A = getattr(material, "A", None) or 1.0
        self.rho_cp = material.rho * material.cp
        self.J = self.L / 2.0  # Jacobian

    def shape_functions(self, r: float) -> np.ndarray:
        return np.array([(1.0 - r) / 2.0, (1.0 + r) / 2.0])

    def compute_B_matrix(self, r: float = 0.0) -> np.ndarray:
        """B = dN/dx"""
        return np.array([[-1.0 / self.L, 1.0 / self.L]])

    def compute_k_e(self) -> np.ndarray:
        """Conductivity matrix K_th = \int{ B^T * k * B * A * dV }
        B is a constant -> use a single point gaussian integration
        """
        points, weights = gauss_1d(1)
        K_th = np.zeros((2, 2))

        for r, w in zip(points, weights):
            B = self.compute_B_matrix(r)  # shape: (1, 2)
            K_th += (B.T @ np.array([[self.k]]) @ B) * self.A * self.J * w
        return K_th

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(2)

    def compute_stress(self, T_e: np.ndarray) -> np.ndarray:
        """Heat flux (Fourier's law): q = -k * B * T_e"""
        B = self.compute_B_matrix()
        # 열유속(q)이 구조 해석의 응력(Stress) 역할을 합니다.
        q = -self.k * (B @ T_e)[0]
        return np.array([q])
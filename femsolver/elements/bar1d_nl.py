from __future__ import annotations

import numpy as np

from femsolver.elements.base import BaseElement
from femsolver.utils.quadrature import gauss_1d


class Bar1DNLElement(BaseElement):
    """1D geometrically nonlinear bar using Gauss Integration.
        Total Lagrangian formulation.
    """

    def __init__(self, element_id, node_ids, nodes_coords, material, **kwargs):
        super().__init__(element_id, node_ids, nodes_coords, material)
        x1 = float(self.nodes_coords[0, 0])
        x2 = float(self.nodes_coords[1, 0])
        self.L = abs(x2 - x1)
        if self.L < 1e-12:
            raise ValueError(f"Element {element_id} is a zero length")

        self.E = material.E
        self.A = material.A
        self.rho_A = material.rho * material.A
        self.J = self.L / 2.0 # Jacobian

    def compute_B_matrix(self) -> np.ndarray:
        return np.array([[-1.0 / self.L, 1.0 / self.L]])

    def shape_functions(self, r: float) -> np.ndarray:
        """ N1 = (1 - r)/2, N2 = (1 + r)/2 for r \in [-1, 1]"""
        return np.array([(1.0 - r) / 2.0, (1.0 + r) / 2.0])


    # Nonlinear Gauss Integration (F_int & K_T)
    def compute_f_int(self, u_e: np.ndarray) -> np.ndarray:
        """Internal force: F_int = \int{ B_NL^T * S * dV0 } (total lagrangian)"""
        points, weights = gauss_1d(1)
        f_int = np.zeros(2)
        B0 = self.compute_B_matrix()

        for r, w in zip(points, weights):
            eps_lin = (B0 @ u_e)[0]
            F_def = 1.0 + eps_lin  # Deformation gradient
            E_GL = eps_lin + 0.5 * eps_lin ** 2  # Green-Lagrange strain
            S = self.E * E_GL  # 2nd PK 응력

            # Update Nonlinear B matrix
            B_NL = F_def * B0

            # integration
            integrand = (B_NL.T * S).flatten()
            f_int += integrand * self.A * self.J * w

        return f_int

    def compute_k_tangent(self, u_e: np.ndarray) -> np.ndarray:
        """Tangent Stiffness: K_T = \int (B_NL^T * E * B_NL + B0^T * S * B0) * dV0"""
        points, weights = gauss_1d(1)
        K_T = np.zeros((2, 2))
        B0 = self.compute_B_matrix()

        for r, w in zip(points, weights):
            eps_lin = (B0 @ u_e)[0]
            F_def = 1.0 + eps_lin
            E_GL = eps_lin + 0.5 * eps_lin ** 2
            S = self.E * E_GL

            B_NL = F_def * B0

            # 1. Material Stiffness
            K_mat = (B_NL.T @ np.array([[self.E]]) @ B_NL) * self.A * self.J * w

            # 2. Geometric Stiffness / Initial Stress
            K_geo = (B0.T @ np.array([[S]]) @ B0) * self.A * self.J * w

            K_T += K_mat + K_geo

        return K_T

    # Standard Linear Methods
    def compute_k_e(self) -> np.ndarray:
        """K_e =  K_T(u = 0)"""
        return self.compute_k_tangent(np.zeros(2))

    def compute_f_e(self) -> np.ndarray:
        return np.zeros(2)

    def compute_stress(self, u_e: np.ndarray) -> np.ndarray:
        B0 = self.compute_B_matrix()
        eps_lin = (B0 @ u_e)[0]
        E_GL = eps_lin + 0.5 * eps_lin ** 2
        return np.array([self.E * E_GL])
from __future__ import annotations

from typing import Optional

import numpy as np

from femsolver.materials.base import BaseMaterial

class LinearElasticMaterial(BaseMaterial):
    def __init__(self,
                 material_id: str,
                 E: float,
                 nu: float,
                 rho: float,
                 A: Optional[float] = None,
                 thickness: Optional[float] = 1.0,
                 alpha: float = 0.0):
        super().__init__(material_id)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self.A = float(A) if A is not None else None
        self.thickness = float(thickness) if thickness is not None else 1.0
        self.alpha = alpha

    def constitutive_matrix(self, plane: str = "stress") -> np.ndarray:
        """Return the 3x3 plane-stress or plane-strain constitutive matrix C."""
        E, nu = self.E, self.nu
        if plane == "stress":
            c = E / (1.0 - nu**2)
            return c * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0-nu) / 2.0]
                ]
            )
        elif plane == "strain":
            c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return c * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu,  0.0],
                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]
                ]
            )
        else:
            raise ValueError(f"Unknown plane condition: {plane}. Use 'stress' or 'strain'")

    def constitutive_matrix_3d(self) -> np.ndarray:
        """ Return the 6x6 isotropic 3D constitutive matrix C.
        Strain order: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
        """
        E, nu = self.E, self.nu
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = (1.0 - 2.0 * nu) / 2.0
        return c*np.array([
            [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
            [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
            [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G],
        ])

    def constitutive_matrix_axisym(self) -> np.ndarray:
        """ Return the 4x4 axisymmetric constitutive matrix C.
        Strain order: [eps_rr, eps_tt, eps_zz, gamma_rz]
        """
        E, nu = self.E, self.nu
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = (1.0 - 2.0 * nu) / 2.0
        return c*np.array([
            [1.0 - nu, nu, nu, 0.0],
            [nu, 1.0 - nu, nu, 0.0],
            [nu, nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.0, G],
        ])

    def bulk_modulus(self) -> float:
        denom = max(3.0 * (1.0 - 2.0 * self.nu), 1.0e-6) # Clamp to avoid zero division
        return self.E / denom


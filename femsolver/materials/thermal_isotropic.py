from __future__ import annotations

from typing import Optional

from femsolver.materials.base import BaseMaterial


class ThermalIsotropicMaterial(BaseMaterial):
    """ Isotropic thermal material

    Attributes:
        k_cond: Thermal conductivity [ W/(m*K) ]
        cp: Specific heat capacity [J/(kg*K)]
        rho:    Density [kg/m^3]
        A:   Cross-sectional area [m^2]
        thickness:  Element thickness [m]
    """

    def __init__(self,
                 material_id: str,
                 k_cond: float,
                 cp: float = 0.0,
                 rho: float = 0.0,
                 A: Optional[float] = None,
                 thickness: Optional[float] = 1.0,
    ):
        super().__init__(material_id)
        self.k_cond = float(k_cond)
        self.cp = float(cp)
        self.rho = float(rho)
        self.A = float(A) if A is not None else None
        self.thickness = float(thickness) if thickness is not None else 1.0

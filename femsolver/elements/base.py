from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from femsolver.materials.base import BaseMaterial


@dataclass
class ElementResult:
    element_id: int
    stress: Optional[np.ndarray] = None
    strain: Optional[np.ndarray] = None
    pressure: Optional[float] = None # for u/p formulation


class BaseElement(ABC):
    def __init__(self,
                 element_id: int,
                 node_ids: List[int],
                 nodes_coords: np.ndarray,
                 material: BaseMaterial):
        self.element_id = element_id
        self.node_ids = node_ids
        self.nodes_coords = np.asarray(nodes_coords, dtype=float)
        self.material = material

    @abstractmethod
    def compute_k_e(self) -> np.ndarray:
        """Element stiffness matrix"""
        ...

    @abstractmethod
    def compute_f_e(self) -> np.ndarray:
        """Element body-force vector"""
        ...

    @abstractmethod
    def compute_stress(selfself, u_e: np.ndarray) -> np.ndarray:
        """Compute element stress from element displacement vector"""
        ...

    def shape_functions(self, *args) -> np.ndarray:
        """Evaluate shape functions at given parametric coordinates."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement shape_functions()"
        )

    def compute_B_matrix(self, *args) -> np.ndarray:
        """Strain-displacement matrix."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_B_matrix()"
        )
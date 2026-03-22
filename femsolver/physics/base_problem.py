from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

@dataclass
class SolveResult:
    problem_name: str
    problem_type: str
    solve_time_s: float = 0.0
    converged: bool = True
    n_dofs: int = 0

    # {node_id: {dof_name: value}} ex) {1: {"ux": 0.0, "uy": 0.0}}
    nodal_displacements: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # {element_id: np.ndarray of stress components}
    element_stresses: Dict[int, np.ndarray] = field(default_factory=dict)

    # {node_id: {force_dof_name: value}} ex) {1: {"fx": -1000.0}}
    reaction_forces: Dict[int, Dict[str, float]] = field(default_factory=dict)


class BaseProblem(ABC):
    @abstractmethod
    def assemble(self) -> None:
        ...

    @abstractmethod
    def apply_boundary_conditions(self) -> None:
        ...

    @abstractmethod
    def solve(self) -> SolveResult:
        ...

    @abstractmethod
    def postprocess(self, u_full: np.ndarray, *args, **kwargs) -> SolveResult: # used by "solve"
        ...
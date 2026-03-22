from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EssentialBC:
    node_id: int
    dof_name: str # ex) ux, uy, ...
    value: float

@dataclass
class NaturalBC:
    """ TODO: a placeholder"""
    pass

@dataclass
class NodalLoad:
    node_id: int
    dof_name: str
    value: float

@dataclass
class LoadCase:
    nodal_loads: List[NodalLoad] = field(default_factory=list)
    natural_bcs: List[NaturalBC] = field(default_factory=list)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

@dataclass
class Node:
    id: int
    coords: np.ndarray

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)

@dataclass
class Element:
    id: int
    type: str
    node_ids: List[int]
    material_id: str

@dataclass
class Mesh:
    nodes: List[Node] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)

    def __post_init__(self):
        self._node_dict: Dict[int, Node] = {n.id: n for n in self.nodes}
        self._elem_dict: Dict[int, Element] = {e.id: e for e in self.elements}

    def get_node(self, node_id: int) -> Node:
        try:
            return self._node_dict[node_id]
        except KeyError:
            raise KeyError(f"Node {node_id} not found in mesh")

    def get_element(self, element_id: int) -> Element:
        try:
            return self._elem_dict[element_id]
        except KeyError:
            raise KeyError(f"Element {element_id} not found in mesh")
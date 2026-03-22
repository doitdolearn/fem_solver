from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from femsolver.core.mesh import Mesh

DOF_NAMES: Dict[int, List[str]] = {
    1: ["ux"],
    2: ["ux", "uy"],
    3: ["ux", "uy", "uz"],
}


class DOFManager:
    """Maps (node_id, dof_name) pairs to global DOF indices"""
    def __init__(self, mesh: Mesh, n_dof_per_node: int, dof_names: Optional[List[str]] = None):
        self.mesh = mesh
        self.n_dof_per_node = n_dof_per_node
        self.dof_names: List[str] = dof_names if dof_names is not None else DOF_NAMES[n_dof_per_node]
        self.n_dofs: int = len(mesh.nodes) * n_dof_per_node
        self._node_dof_map: Dict[int, Dict[str, int]] = {}
        self._build_map()

    def _build_map(self) -> None:
        for i, node in enumerate(self.mesh.nodes):
            self._node_dof_map[node.id] = {
                name: i * self.n_dof_per_node + j
                for j, name in enumerate(self.dof_names)
            }

    def get_global_dof(self, node_id, dof_name: str) -> int:
        try:
            return self._node_dof_map[node_id][dof_name]
        except KeyError:
            raise KeyError(
                f"DOF '{dof_name}' not found for node {node_id}."
                f"Available DOFs: {self.dof_names}"
            )

    def get_element_dofs(self, node_ids: List[int]) -> List[int]:
        """Return ordered global DOF indices for all nodes of an element. """
        dofs = []
        for node_id in node_ids:
            for name in self.dof_names:
                dofs.append(self._node_dof_map[node_id][name])
        return dofs

    def partition_dofs(self, constrained: Dict[int, List[str]]) -> Tuple[List[int], List[int]]:
        """
        Args:
            constrained: {node_id: [dof_name, ...]} (Mapping of constrained DOFs)

        Return (free_dofs, constrained_dofs)
        """
        constrained_set: set = set()
        for node_id, dof_names in constrained.items():
            for name in dof_names:
                constrained_set.add(self.get_global_dof(node_id, name))
        free = [i for i in range(self.n_dofs) if i not in constrained_set]
        constrained_list = sorted(constrained_set)
        return free, constrained_list

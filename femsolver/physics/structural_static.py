from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from femsolver.core.assembler import Assembler
from femsolver.core.boundary_conditions import EssentialBC, LoadCase
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Mesh
from femsolver.elements.base import BaseElement
from femsolver.physics.base_problem import BaseProblem, SolveResult
from femsolver.postprocessing.reaction_forces import compute_reactions
from femsolver.solvers.direct import DirectSolver
from femsolver.utils.linalg_utils import partition_system

# Maps displacement DOF names to force output names
_DISP_TO_FORCE = {"ux": "fx", "uy": "fy", "uz": "fz", "ur": "fr", "utheta": "ftheta"}

class StructuralStaticProblem(BaseProblem):
    def __init__(self,
                 mesh: Mesh,
                 dof_manager: DOFManager,
                 elements: List[BaseElement],
                 essential_bcs: List[EssentialBC],
                 load_case: LoadCase,
                 material_map: Dict,
                 problem_name: str = "unnamed",
                 solver=None):
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.elements = elements
        self.essential_bcs = essential_bcs
        self.load_case = load_case
        self.material_map = material_map
        self.problem_name = problem_name
        self.solver = solver or DirectSolver()

        self.K_global = None
        self.F_global = None

    def assemble(self) -> None:
        assembler = Assembler(self.mesh, self.dof_manager, self.elements)
        self.K_global, self.F_global = assembler.assemble_K_and_F()

        # Add nodal loads to the global force vector
        for load in self.load_case.nodal_loads:
            global_dof = self.dof_manager.get_global_dof(load.node_id, load.dof_name)
            self.F_global[global_dof] += load.value

    def apply_boundary_conditions(self) -> None:
        """ Partition method is used -> BCs are applied inside the solve()"""
        pass

    def solve(self) -> SolveResult:
        t0 = time.perf_counter()
        self.assemble()

        constrained_values: Dict[int, float] = {}
        constrained_by_node: Dict[int, List[str]] = {}
        for bc in self.essential_bcs:
            global_dof = self.dof_manager.get_global_dof(bc.node_id, bc.dof_name)
            constrained_values[global_dof] = bc.value
            constrained_by_node.setdefault(bc.node_id, []).append(bc.dof_name)

        free_dofs, constrained_dofs = self.dof_manager.partition_dofs(constrained_by_node)

        u_full = partition_system(
            self.K_global,
            self.F_global,
            free_dofs,
            constrained_dofs,
            constrained_values,
            solve_fn = self.solver.solve,
        )

        dt = time.perf_counter() - t0
        return self.postprocess(u_full, self.F_global, dt)

    def postprocess(self,
                    u_full: np.ndarray,
                    F_ext: Optional[np.ndarray] = None,
                    solve_time: float = 0.0) -> SolveResult:
        dof_names = self.dof_manager.dof_names

        # Nodal displacements
        nodal_displacements: Dict[int, Dict[str, float]] = {}
        for node in self.mesh.nodes:
            displacements = {}
            for name in dof_names:
                global_dof = self.dof_manager.get_global_dof(node.id, name)
                displacements[name] = float(u_full[global_dof])
            nodal_displacements[node.id] = displacements

        # Element stresses
        element_stresses: Dict[int, np.ndarray] = {}
        for element in self.elements:
            dofs = self.dof_manager.get_element_dofs(element.node_ids)
            u_e = u_full[dofs]
            element_stresses[element.element_id] = element.compute_stress(u_e)

        # Reaction forces
        reaction_forces: Dict[int, Dict[str, float]] = {}
        if self.K_global is not None and F_ext is not None:
            R_full = compute_reactions(self.K_global, u_full, F_ext)
            for bc in self.essential_bcs:
                node_id = bc.node_id
                global_dof = self.dof_manager.get_global_dof(node_id, bc.dof_name)
                force_name = _DISP_TO_FORCE.get(bc.dof_name, bc.dof_name)
                reaction_forces.setdefault(node_id, {})[force_name] = float(R_full[global_dof])

        return SolveResult(
            problem_name=self.problem_name,
            problem_type="structural_static",
            solve_time_s=solve_time,
            converged=True,
            n_dofs=self.dof_manager.n_dofs,
            nodal_displacements=nodal_displacements,
            element_stresses=element_stresses,
            reaction_forces=reaction_forces,
        )




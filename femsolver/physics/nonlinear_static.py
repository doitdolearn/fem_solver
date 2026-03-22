from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import coo_matrix

from femsolver.core.assembler import Assembler
from femsolver.core.boundary_conditions import EssentialBC, LoadCase
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Mesh
from femsolver.elements.base import BaseElement
from femsolver.physics.base_problem import BaseProblem, SolveResult
from femsolver.solvers.newton_raphson import NewtonRaphsonSolver

# Maps displacement DOF names to reaction force output names
_DISP_TO_FORCE: Dict[str, str] = {"ux": "fx", "uy": "fy", "uz": "fz"}


class NonlinearStaticProblem(BaseProblem):
    """Geometrically nonlinear static analysis via load stepping + Newton-Raphson.

    The total external load is applied in "n_load_steps" equal increments.
    At each step, Newton-Raphson iterations find the displacement that satisfies:
        F_int(u) = \lambda * F_ext
    """

    def __init__(
        self,
        mesh: Mesh,
        dof_manager: DOFManager,
        elements: List[BaseElement],
        essential_bcs: List[EssentialBC],
        load_case: LoadCase,
        material_map: Dict,
        problem_name: str = "unnamed",
        n_load_steps: int = 10,
        solver: Optional[NewtonRaphsonSolver] = None,
    ):
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.elements = elements
        self.essential_bcs = essential_bcs
        self.load_case = load_case
        self.material_map = material_map
        self.problem_name = problem_name
        self.n_load_steps = n_load_steps
        self.solver = solver or NewtonRaphsonSolver()

    # Nonlinear assembly
    def _assemble_KT_and_Fint(self, u_full: np.ndarray):
        """Assemble the tangent stiffness and internal force from "u_full" """
        n = self.dof_manager.n_dofs
        rows, cols, vals = [], [], []
        F_int = np.zeros(n)

        for elem in self.elements:
            dofs = self.dof_manager.get_element_dofs(elem.node_ids)
            u_e = u_full[dofs]

            if hasattr(elem, "compute_k_tangent"):
                K_e = elem.compute_k_tangent(u_e)
                f_e = elem.compute_f_int(u_e)
            else:
                # Linear element
                K_e = elem.compute_k_e()
                f_e = K_e @ u_e

            for i, di in enumerate(dofs):
                F_int[di] += f_e[i]
                for j, dj in enumerate(dofs):
                    rows.append(di)
                    cols.append(dj)
                    vals.append(K_e[i, j])

        K_T = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        return K_T, F_int

    def assemble(self) -> None:
        pass  # Assembly is performed inside solve() at each NR iteration.

    def apply_boundary_conditions(self) -> None:
        pass

    def solve(self) -> SolveResult:
        t0 = time.perf_counter()
        n_dofs = self.dof_manager.n_dofs

        # Build constrained DOF maps (same to the one in structural static)
        constrained_values: Dict[int, float] = {}
        constrained_by_node: Dict[int, List[str]] = {}
        for bc in self.essential_bcs:
            global_dof = self.dof_manager.get_global_dof(bc.node_id, bc.dof_name)
            constrained_values[global_dof] = bc.value
            constrained_by_node.setdefault(bc.node_id, []).append(bc.dof_name)

        free_dofs, constrained_dofs = self.dof_manager.partition_dofs(constrained_by_node)
        u_c = np.array([constrained_values.get(d, 0.0) for d in constrained_dofs])

        # Build total external force vector
        F_ext = np.zeros(n_dofs)
        for load in self.load_case.nodal_loads:
            global_dof = self.dof_manager.get_global_dof(load.node_id, load.dof_name)
            F_ext[global_dof] += load.value

        # Initial displacement (prescribed BCs applied, others zero)
        u_full = np.zeros(n_dofs)
        u_full[constrained_dofs] = u_c

        # Proportional load stepping
        converged_all = True
        for step in range(1, self.n_load_steps + 1):
            lam = step / self.n_load_steps
            F_step = lam * F_ext
            F_ext_f = F_step[free_dofs]

            def _compute_KT_and_R(u_f: np.ndarray, _u_full=u_full):
                u_tmp = _u_full.copy()
                u_tmp[free_dofs] = u_f
                K_T, F_int = self._assemble_KT_and_Fint(u_tmp)
                K_T_ff = K_T[free_dofs, :][:, free_dofs]
                R_f = F_int[free_dofs] - F_ext_f
                return K_T_ff, R_f

            u_f_init = u_full[free_dofs].copy()
            u_f, _, step_converged = self.solver.solve(_compute_KT_and_R, u_f_init, F_ext_f)
            u_full[free_dofs] = u_f

            if not step_converged:
                converged_all = False
                break
        dt = time.perf_counter() - t0
        return self.postprocess(u_full, F_ext, dt, converged_all)

    def postprocess(
        self,
        u_full: np.ndarray,
        F_ext: Optional[np.ndarray] = None,
        solve_time: float = 0.0,
        converged: bool = True,
    ) -> SolveResult:
        dof_names = self.dof_manager.dof_names

        # Nodal displacements
        nodal_displacements: Dict[int, Dict[str, float]] = {}
        for node in self.mesh.nodes:
            disps = {}
            for name in dof_names:
                gdof = self.dof_manager.get_global_dof(node.id, name)
                disps[name] = float(u_full[gdof])
            nodal_displacements[node.id] = disps

        # Element stresses (2nd PK stress for bar1d_nl)
        element_stresses: Dict[int, np.ndarray] = {}
        for elem in self.elements:
            dofs = self.dof_manager.get_element_dofs(elem.node_ids)
            u_e = u_full[dofs]
            element_stresses[elem.element_id] = elem.compute_stress(u_e)

        # Reaction forces: R = F_int(u) - F_ext at constrained DOFs
        reaction_forces: Dict[int, Dict[str, float]] = {}
        if F_ext is not None:
            _, F_int_final = self._assemble_KT_and_Fint(u_full)
            R_full = F_int_final - F_ext
            for bc in self.essential_bcs:
                node_id = bc.node_id
                gdof = self.dof_manager.get_global_dof(node_id, bc.dof_name)
                force_name = _DISP_TO_FORCE.get(bc.dof_name, bc.dof_name)
                reaction_forces.setdefault(node_id, {})[force_name] = float(
                    R_full[gdof]
                )

        return SolveResult(
            problem_name=self.problem_name,
            problem_type="nonlinear_static",
            solve_time_s=solve_time,
            converged=converged,
            n_dofs=self.dof_manager.n_dofs,
            nodal_displacements=nodal_displacements,
            element_stresses=element_stresses,
            reaction_forces=reaction_forces,
        )


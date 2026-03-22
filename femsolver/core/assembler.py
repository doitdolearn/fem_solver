from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Mesh
from femsolver.elements.base import BaseElement

class Assembler:
    """Assemble global K, F matrices with COO -> CSR"""
    def __init__(self, mesh: Mesh, dof_manager: DOFManager, elements: List[BaseElement]):
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.elements = elements
        self.n_dofs = dof_manager.n_dofs

    def assemble_K_and_F(self) -> Tuple[csr_matrix, np.ndarray]:
        rows, cols, vals = [], [], []
        F = np.zeros(self.n_dofs)

        for elem in self.elements:
            k_e = elem.compute_k_e()
            f_e = elem.compute_f_e()
            dofs = self.dof_manager.get_element_dofs(elem.node_ids)

            for i, di in enumerate(dofs):
                F[di] += f_e[i]
                for j, dj in enumerate(dofs):
                    rows.append(di)
                    cols.append(dj)
                    vals.append(k_e[i, j])

        K = coo_matrix(
            (vals, (rows, cols)), shape=(self.n_dofs, self.n_dofs)
        ).tocsr()

        return K, F

"""Solving tests for the 1D bar/truss pipeline.

These tests do not use yaml io codes.
"""

import numpy as np
import pytest

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.bar1d import Bar1DElement
from femsolver.materials.linear_elastic import LinearElasticMaterial
from femsolver.physics.structural_static import StructuralStaticProblem


def _build_chain(n_elements: int, E=200e9, A=0.01, L=1.0, rho=7850.0):
    """Build an n-element chain of 1D bars, fixed at node 1."""
    mat = LinearElasticMaterial("steel", E=E, nu=0.3, rho=rho, A=A)
    n_nodes = n_elements + 1
    node_L = L / n_elements  # length per element

    nodes = [Node(i + 1, np.array([i * node_L])) for i in range(n_nodes)]
    mesh_elems = [
        Element(i + 1, "bar1d", [i + 1, i + 2], "steel") for i in range(n_elements)
    ]
    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dof_mgr = DOFManager(mesh, n_dof_per_node=1)
    fem_elems = [
        Bar1DElement(
            i + 1,
            [i + 1, i + 2],
            np.array([[i * node_L], [(i + 1) * node_L]]),
            mat,
        )
        for i in range(n_elements)
    ]
    mat_map = {"steel": mat}
    return mesh, dof_mgr, fem_elems, mat_map


def _solve_chain(n_elements: int, P: float, E=200e9, A=0.01, L=1.0):
    """Fixed-free chain under tip load P. Returns SolveResult."""
    mesh, dof_mgr, elems, mat_map = _build_chain(n_elements, E=E, A=A, L=L)
    n_tip = n_elements + 1
    bcs = [EssentialBC(node_id=1, dof_name="ux", value=0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(node_id=n_tip, dof_name="ux", value=P)])
    prob = StructuralStaticProblem(mesh, dof_mgr, elems, bcs, loads, mat_map)
    return prob.solve()


# ----- Single bar -----

def test_single_bar_tip_displacement():
    """u_tip = PL / (EA), exact for any number of elements."""
    E, A, L, P = 200e9, 0.01, 1.0, 1000.0
    result = _solve_chain(1, P, E=E, A=A, L=L)
    u_tip = result.nodal_displacements[2]["ux"]
    expected = P * L / (E * A)
    np.testing.assert_allclose(u_tip, expected, rtol=1e-10)


def test_single_bar_fixed_node_zero_displacement():
    result = _solve_chain(1, P=1000.0)
    assert result.nodal_displacements[1]["ux"] == pytest.approx(0.0, abs=1e-30)


def test_single_bar_reaction_equals_negative_load():
    """Reaction at fixed node = -P (Newton's third law)."""
    P = 1000.0
    result = _solve_chain(1, P)
    rx = result.reaction_forces[1]["fx"]
    np.testing.assert_allclose(rx, -P, rtol=1e-10)


def test_single_bar_stress():
    """σ = P / A for a bar under axial load."""
    E, A, L, P = 200e9, 0.01, 1.0, 1000.0
    result = _solve_chain(1, P, E=E, A=A, L=L)
    stress = result.element_stresses[1][0]
    np.testing.assert_allclose(stress, P / A, rtol=1e-10)


# ----- Multi-element chain -----

@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_n_bar_tip_displacement(n):
    """u_tip = PL/(EA) is exact for any mesh refinement."""
    E, A, L, P = 200e9, 0.01, 1.0, 1000.0
    result = _solve_chain(n, P, E=E, A=A, L=L)
    u_tip = result.nodal_displacements[n + 1]["ux"]
    expected = P * L / (E * A)
    np.testing.assert_allclose(u_tip, expected, rtol=1e-9)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_reactions_sum_to_load(n):
    """Sum of all reaction forces must equal the applied load."""
    P = 500.0
    result = _solve_chain(n, P)
    total_rx = sum(
        forces.get("fx", 0.0) for forces in result.reaction_forces.values()
    )
    np.testing.assert_allclose(total_rx + P, 0.0, atol=1e-6)


# ----- Non-zero prescribed displacement -----

def test_prescribed_nonzero_displacement():
    """Fixed-fixed bar: prescribe u1=0, u2=δ. Stress = E*δ/L."""
    E, A, L, delta = 100.0, 1.0, 1.0, 0.01
    mat = LinearElasticMaterial("m", E=E, nu=0.3, rho=1.0, A=A)
    nodes = [Node(1, np.array([0.0])), Node(2, np.array([L]))]
    mesh = Mesh(nodes=nodes, elements=[Element(1, "bar1d", [1, 2], "m")])
    dof_mgr = DOFManager(mesh, n_dof_per_node=1)
    elems = [Bar1DElement(1, [1, 2], np.array([[0.0], [L]]), mat)]
    bcs = [
        EssentialBC(node_id=1, dof_name="ux", value=0.0),
        EssentialBC(node_id=2, dof_name="ux", value=delta),
    ]
    loads = LoadCase(nodal_loads=[])
    prob = StructuralStaticProblem(mesh, dof_mgr, elems, bcs, loads, {"m": mat})
    result = prob.solve()

    stress = result.element_stresses[1][0]
    np.testing.assert_allclose(stress, E * delta / L, rtol=1e-12)

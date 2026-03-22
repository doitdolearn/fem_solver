"""Integration tests for geometrically nonlinear static analysis.

Total Lagrangian 1D bar — exact analytical solution
====================================================
For a bar of initial length L₀, area A₀, Young's modulus E, fixed at node 1,
with axial force P applied at node 2:

    Equilibrium:  N(u₂) = P
    N = A₀·(1 + ε)·S  where  S = E·E_GL,  E_GL = ε + ε²/2,  ε = u₂/L₀

Expanding:
    E·A₀·(ε + 3ε²/2 + ε³/2) = P

For E = A₀ = L₀ = 1 and ε = 0.5 (u₂ = 0.5):
    N = 1·(1.5)·(0.5 + 0.125) = 1.5·0.625 = 0.9375  → P = 0.9375

Key comparisons
---------------
Linear FEM for P = 0.9375:   u₂_lin = P/(E·A/L) = 0.9375  (87.5 % over-estimate!)
Nonlinear FEM:                u₂_nl  = 0.5                 (exact)
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.bar1d_nl import Bar1DNLElement
from femsolver.materials.linear_elastic import LinearElasticMaterial
from femsolver.physics.nonlinear_static import NonlinearStaticProblem
from femsolver.solvers.newton_raphson import NewtonRaphsonSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(E=1.0, A=1.0, L=1.0, rho=0.0):
    return LinearElasticMaterial("m", E=E, nu=0.0, rho=rho, A=A)


def build_single_bar_nl(P: float, E=1.0, A=1.0, L=1.0, n_steps=10, tol=1e-12):
    """Single bar1d_nl: fixed at node 1, force P at node 2."""
    mat = _bar(E=E, A=A, L=L)
    nodes = [Node(1, np.array([0.0])), Node(2, np.array([L]))]
    mesh = Mesh(
        nodes=nodes,
        elements=[Element(1, "bar1d_nl", [1, 2], "m")],
    )
    dof_manager = DOFManager(mesh, n_dof_per_node=1)
    elem = Bar1DNLElement(1, [1, 2], np.array([[0.0], [L]]), mat)
    bcs = [EssentialBC(1, "ux", 0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(2, "ux", P)])
    return NonlinearStaticProblem(
        mesh, dof_manager, [elem], bcs, loads, {},
        n_load_steps=n_steps, solver=NewtonRaphsonSolver(tol=tol),
    )


def build_multi_bar_nl(n: int, P: float, E=1.0, A=1.0, L=1.0, n_steps=20):
    """n bar1d_nl elements in series; fixed at left, force P at right."""
    mat = _bar(E=E, A=A, L=L / n)
    dx = L / n
    nodes = [Node(i + 1, np.array([i * dx])) for i in range(n + 1)]
    mesh_elems = []
    elements = []
    for i in range(n):
        nids = [i + 1, i + 2]
        coords = np.array([[i * dx], [(i + 1) * dx]])
        mesh_elems.append(Element(i + 1, "bar1d_nl", nids, "m"))
        elements.append(Bar1DNLElement(i + 1, nids, coords, mat))
    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dof_manager = DOFManager(mesh, n_dof_per_node=1)
    bcs = [EssentialBC(1, "ux", 0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(n + 1, "ux", P)])
    return NonlinearStaticProblem(
        mesh, dof_manager, elements, bcs, loads, {}, n_load_steps=n_steps
    )


# ---------------------------------------------------------------------------
# Tangent stiffness & internal force tests
# ---------------------------------------------------------------------------

def test_bar1d_nl_tangent_at_zero_equals_linear():
    """At zero displacement, K_T must equal the linear stiffness."""
    mat = _bar(E=200e9, A=0.01, L=1.0)
    elem = Bar1DNLElement(1, [1, 2], np.array([[0.0], [1.0]]), mat)
    u_zero = np.zeros(2)
    K_T = elem.compute_k_tangent(u_zero)
    K_lin = elem.compute_k_e()
    np.testing.assert_allclose(K_T, K_lin, rtol=1e-14)


def test_bar1d_nl_internal_force_at_zero():
    """F_int must be zero at zero displacement."""
    mat = _bar()
    elem = Bar1DNLElement(1, [1, 2], np.array([[0.0], [1.0]]), mat)
    np.testing.assert_allclose(elem.compute_f_int(np.zeros(2)), np.zeros(2), atol=1e-14)


def test_bar1d_nl_internal_force_known_displacement():
    """F_int = N = A·(1+ε)·S for given u₂."""
    E, A, L = 1.0, 1.0, 1.0
    mat = _bar(E=E, A=A, L=L)
    elem = Bar1DNLElement(1, [1, 2], np.array([[0.0], [L]]), mat)
    u_e = np.array([0.0, 0.5])       # ε = 0.5
    eps = 0.5
    E_GL = eps + 0.5 * eps ** 2      # 0.625
    S = E * E_GL                     # 0.625
    N_expected = A * (1.0 + eps) * S # 0.9375
    f_int = elem.compute_f_int(u_e)
    np.testing.assert_allclose(f_int[1], N_expected, rtol=1e-12)
    np.testing.assert_allclose(f_int[0], -N_expected, rtol=1e-12)


def test_bar1d_nl_tangent_symmetric():
    """K_T must be symmetric for all displacement values."""
    mat = _bar(E=200e9, A=0.01, L=2.0)
    elem = Bar1DNLElement(1, [1, 2], np.array([[0.0], [2.0]]), mat)
    for u2 in [0.0, 0.1, 0.5, -0.05]:
        K_T = elem.compute_k_tangent(np.array([0.0, u2]))
        np.testing.assert_allclose(K_T, K_T.T, atol=1e-14)


# ---------------------------------------------------------------------------
# Nonlinear static solve tests
# ---------------------------------------------------------------------------

def test_single_bar_nl_small_load_matches_linear():
    """For small P, nonlinear solution ≈ linear solution."""
    E, A, L = 200e9, 0.01, 1.0
    P = 1.0   # tiny load relative to EA = 2e9
    result = build_single_bar_nl(P=P, E=E, A=A, L=L).solve()
    u_nl = result.nodal_displacements[2]["ux"]
    u_lin = P * L / (E * A)
    np.testing.assert_allclose(u_nl, u_lin, rtol=1e-6)


def test_single_bar_nl_analytical_large_strain():
    """NR must recover the analytical solution for P=0.9375 (ε=0.5 exactly).

    Linear FEM gives u₂ = 0.9375 (87.5 % over-estimate).
    Nonlinear FEM must give u₂ = 0.5.
    """
    P = 0.9375   # E=A=L=1, analytical solution: ε=0.5, u₂=0.5
    result = build_single_bar_nl(P=P, n_steps=20).solve()
    u_nl = result.nodal_displacements[2]["ux"]
    np.testing.assert_allclose(u_nl, 0.5, rtol=1e-8,
                                err_msg=f"u₂={u_nl:.6f}, expected 0.5")
    assert result.converged


def test_single_bar_nl_reaction_balances_load():
    """Reaction force at the fixed node must equal the applied load."""
    P = 0.9375
    result = build_single_bar_nl(P=P, n_steps=20).solve()
    R = result.reaction_forces[1]["fx"]
    # Equilibrium: R + P = 0 → R = -P
    np.testing.assert_allclose(R, -P, rtol=1e-8)


def test_multi_bar_nl_convergence():
    """Multi-element bar should converge to the same analytical solution."""
    P = 0.9375
    # For n elements each of length L/n with same E,A: total elongation = 0.5*L
    result = build_multi_bar_nl(n=5, P=P).solve()
    u_tip = result.nodal_displacements[6]["ux"]
    np.testing.assert_allclose(u_tip, 0.5, rtol=1e-6)


def test_nonlinear_gives_smaller_displacement_than_linear():
    """Geometric stiffening: nonlinear tip displacement < linear for tension."""
    P = 0.9375
    result = build_single_bar_nl(P=P, n_steps=20).solve()
    u_nl = result.nodal_displacements[2]["ux"]
    u_lin = P  # E=A=L=1
    assert u_nl < u_lin, (
        f"Nonlinear u={u_nl:.4f} should be less than linear u={u_lin:.4f}"
    )


def test_nr_converged_flag():
    """Result.converged must be True for a well-posed problem."""
    result = build_single_bar_nl(P=0.5, n_steps=10).solve()
    assert result.converged


def test_nr_problem_type():
    """Result problem_type must be 'nonlinear_static'."""
    result = build_single_bar_nl(P=0.1).solve()
    assert result.problem_type == "nonlinear_static"

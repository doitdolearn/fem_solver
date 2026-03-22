"""Integration tests for thermal steady-state analysis.

1D Rod:
  - k=50 W/(m·K), A=0.01 m², L=1 m
  - T(0) = 0 K, T(L) = 100 K  (prescribed temperatures)
  - Exact: T(x) = 100·x/L   (linear, no internal source)
  - Heat flux: q = -k·(T2-T1)/L = -5000 W/m² (negative → flows toward x=0)
  - Reactions: Q_left = -k·A·(T2-T1)/L = 50 [W] (into left node)
               Q_right = +k·A·(T2-T1)/L = -50 [W] (out of right node)
               Sum = 0 (heat balance)

2D Patch (Tri3Thermal):
  - Unit square divided into 2 CST triangles
  - T(0,y) = 0, T(1,y) = 100 → T(x,y) = 100·x
  - Heat flux: qx = -k·100, qy = 0
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.rod1d_thermal import Rod1DThermalElement
from femsolver.elements.tri3_thermal import Tri3ThermalElement
from femsolver.materials.thermal_isotropic import ThermalIsotropicMaterial
from femsolver.physics.thermal_steady import ThermalSteadyProblem


# ---------------------------------------------------------------------------
# Helper: build a 1D thermal rod problem
# ---------------------------------------------------------------------------

def build_thermal_rod(
    n: int = 4,
    T_left: float = 0.0,
    T_right: float = 100.0,
    k: float = 50.0,
    A: float = 0.01,
    L: float = 1.0,
    internal_Q: float = 0.0,  # uniform nodal heat source at interior nodes [W]
):
    """Build an n-element 1D thermal rod from x=0 to x=L."""
    mat = ThermalIsotropicMaterial("rod", k_cond=k, cp=0.0, rho=0.0, A=A)
    dx = L / n
    nodes = [Node(i + 1, np.array([i * dx])) for i in range(n + 1)]
    mesh_elems = []
    elements = []
    for i in range(n):
        nids = [i + 1, i + 2]
        coords = np.array([[i * dx], [(i + 1) * dx]])
        mesh_elems.append(Element(i + 1, "rod1d_thermal", nids, "rod"))
        elements.append(Rod1DThermalElement(i + 1, nids, coords, mat))

    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dof_manager = DOFManager(mesh, n_dof_per_node=1, dof_names=["T"])

    bcs = [
        EssentialBC(1, "T", T_left),
        EssentialBC(n + 1, "T", T_right),
    ]

    nodal_loads = []
    if internal_Q != 0.0:
        for i in range(1, n):  # interior nodes (0-indexed: 1..n-1)
            nodal_loads.append(NodalLoad(i + 1, "T", internal_Q))

    load_case = LoadCase(nodal_loads=nodal_loads)
    return ThermalSteadyProblem(mesh, dof_manager, elements, bcs, load_case, {})


# ---------------------------------------------------------------------------
# 1D Rod Tests
# ---------------------------------------------------------------------------

def test_1d_rod_linear_temperature():
    """T(0)=0, T(L)=100, no source → T(x) = 100·x/L exactly at all nodes."""
    n, L, T1, T2 = 10, 1.0, 0.0, 100.0
    result = build_thermal_rod(n=n, T_left=T1, T_right=T2, L=L).solve()
    dx = L / n
    for i in range(n + 1):
        T_expected = T1 + (T2 - T1) * i / n
        T_computed = result.nodal_displacements[i + 1]["T"]
        np.testing.assert_allclose(
            T_computed, T_expected, atol=1e-10,
            err_msg=f"Node {i+1} at x={i*dx:.2f}: expected T={T_expected:.2f}"
        )


def test_1d_rod_reaction_heat_balance():
    """Net heat flow at constrained nodes = 0 (no internal source)."""
    result = build_thermal_rod(n=4, T_left=0.0, T_right=100.0).solve()
    Q_total = sum(
        r.get("Q", 0.0) for r in result.reaction_forces.values()
    )
    np.testing.assert_allclose(Q_total, 0.0, atol=1e-8)


def test_1d_rod_constant_heat_flux():
    """For linear T, heat flux q = -k*(T_right-T_left)/L constant in all elements."""
    k, L, T1, T2 = 50.0, 1.0, 0.0, 100.0
    result = build_thermal_rod(n=4, T_left=T1, T_right=T2, k=k, L=L).solve()
    q_expected = -k * (T2 - T1) / L  # = -5000 W/m²
    for eid, flux in result.element_stresses.items():
        np.testing.assert_allclose(
            flux[0], q_expected, rtol=1e-10,
            err_msg=f"Element {eid}: q={flux[0]:.4f} ≠ {q_expected:.4f}"
        )


def test_1d_rod_reaction_magnitudes():
    """Reaction at constrained nodes = ±k·A·(T2-T1)/L."""
    k, A, L, T1, T2 = 50.0, 0.01, 1.0, 0.0, 100.0
    result = build_thermal_rod(n=4, T_left=T1, T_right=T2, k=k, A=A, L=L).solve()
    Q_expected = k * A * (T2 - T1) / L  # 50 W

    # Node 1 (T=0, cold end): external source must extract heat to hold T=0
    # → reaction R = K·T - F_ext = k·A/L·(T1-T2) = -50 W  (negative = extraction)
    Q1 = result.reaction_forces[1]["Q"]
    np.testing.assert_allclose(Q1, -Q_expected, rtol=1e-10)

    # Node 5 (T=100, hot end): external source must supply heat to hold T=100
    # → reaction R = k·A/L·(T2-T1) = +50 W  (positive = injection)
    Q_last = result.reaction_forces[5]["Q"]
    np.testing.assert_allclose(Q_last, Q_expected, rtol=1e-10)


def test_1d_rod_different_temperatures():
    """Vary boundary temperatures and check linearity."""
    for T1, T2 in [(20.0, 80.0), (-10.0, 10.0), (100.0, 0.0)]:
        result = build_thermal_rod(n=6, T_left=T1, T_right=T2, L=1.0).solve()
        for i in range(7):
            T_exp = T1 + (T2 - T1) * i / 6
            T_got = result.nodal_displacements[i + 1]["T"]
            np.testing.assert_allclose(T_got, T_exp, atol=1e-10)


# ---------------------------------------------------------------------------
# 2D Tri3Thermal Patch Test
# ---------------------------------------------------------------------------

def test_tri3_thermal_patch_test():
    """Two CST triangles on unit square; T = 100·x → constant flux."""
    k = 10.0
    mat = ThermalIsotropicMaterial("m", k_cond=k, cp=0.0, rho=0.0, thickness=1.0)

    # Nodes: 1(0,0), 2(1,0), 3(1,1), 4(0,1)
    nodes = [
        Node(1, np.array([0.0, 0.0])),
        Node(2, np.array([1.0, 0.0])),
        Node(3, np.array([1.0, 1.0])),
        Node(4, np.array([0.0, 1.0])),
    ]
    mesh = Mesh(
        nodes=nodes,
        elements=[
            Element(1, "tri3_thermal", [1, 2, 3], "m"),
            Element(2, "tri3_thermal", [1, 3, 4], "m"),
        ],
    )
    dof_manager = DOFManager(mesh, n_dof_per_node=1, dof_names=["T"])
    elem1 = Tri3ThermalElement(
        1, [1, 2, 3], np.array([[0, 0], [1, 0], [1, 1]], dtype=float), mat
    )
    elem2 = Tri3ThermalElement(
        2, [1, 3, 4], np.array([[0, 0], [1, 1], [0, 1]], dtype=float), mat
    )

    # Prescribe T = 100*x at all nodes (all constrained → no free DOFs)
    bcs = [
        EssentialBC(1, "T", 0.0),   # (0,0)
        EssentialBC(2, "T", 100.0), # (1,0)
        EssentialBC(3, "T", 100.0), # (1,1)
        EssentialBC(4, "T", 0.0),   # (0,1)
    ]
    prob = ThermalSteadyProblem(
        mesh, dof_manager, [elem1, elem2], bcs, LoadCase(nodal_loads=[]), {}
    )
    result = prob.solve()

    # Heat flux: qx = -k * dT/dx = -k * 100 = -1000 W/m², qy = 0
    qx_expected = -k * 100.0
    for eid in [1, 2]:
        q = result.element_stresses[eid]
        np.testing.assert_allclose(q[0], qx_expected, rtol=1e-10,
                                   err_msg=f"Element {eid}: qx wrong")
        np.testing.assert_allclose(q[1], 0.0, atol=1e-10,
                                   err_msg=f"Element {eid}: qy should be 0")


def test_tri3_thermal_stiffness_symmetric():
    """Tri3ThermalElement k_e should be symmetric."""
    mat = ThermalIsotropicMaterial("m", k_cond=25.0, cp=0.0, rho=0.0, thickness=0.01)
    coords = np.array([[0, 0], [1, 0], [0.5, 1.0]], dtype=float)
    elem = Tri3ThermalElement(1, [1, 2, 3], coords, mat)
    K = elem.compute_k_e()
    np.testing.assert_allclose(K, K.T, atol=1e-14)

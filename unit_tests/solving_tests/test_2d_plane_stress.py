"""Integration tests for 2D plane-stress analysis.

Cook's membrane benchmark (Cook, 1974):
  - Tapered panel; corners at (0,0), (48,44), (48,60), (0,44).
  - E=1, ν=1/3, plane stress, thickness=1.
  - Left edge (x=0) fully fixed.
  - Unit shear load (F_y=1) uniformly distributed on right edge (x=48).
  - Reference tip displacement v_C at (48,60): ~ 23.97 (converged).
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.quad4 import Quad4Element
from femsolver.elements.tri3 import Tri3Element
from femsolver.materials.linear_elastic import LinearElasticMaterial
from femsolver.physics.structural_static import StructuralStaticProblem


# ---------------------------------------------------------------------------
# Cook's membrane helpers
# ---------------------------------------------------------------------------

def _cooks_node_coords(n: int):
    """Return dict {node_id: (x, y)} for an (n+1)×(n+1) node grid."""
    # Bilinear map: x=48*s, y=44*s + 44*t - 28*s*t
    nodes = {}
    for j in range(n + 1):
        for i in range(n + 1):
            s = i / n
            t = j / n
            x = 48.0 * s
            y = 44.0 * s + 44.0 * t - 28.0 * s * t
            nid = j * (n + 1) + i + 1
            nodes[nid] = (x, y)
    return nodes


def build_cooks_membrane(n: int):
    """Build the Cook's membrane problem on an n×n Q4 mesh.

    Returns a StructuralStaticProblem ready to solve.
    """
    mat = LinearElasticMaterial("m", E=1.0, nu=1.0 / 3.0, rho=0.0, thickness=1.0)
    node_coords = _cooks_node_coords(n)

    nodes = [Node(nid, np.array(xy)) for nid, xy in sorted(node_coords.items())]
    mesh_elems = []
    elements = []
    eid = 1
    for ej in range(n):
        for ei in range(n):
            bl = ej * (n + 1) + ei + 1
            br = bl + 1
            tr = br + (n + 1)
            tl = bl + (n + 1)
            node_ids = [bl, br, tr, tl]
            coords = np.array([node_coords[nid] for nid in node_ids])
            mesh_elems.append(Element(eid, "quad4", node_ids, "m"))
            elements.append(Quad4Element(eid, node_ids, coords, mat, plane="stress"))
            eid += 1

    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dof_manager = DOFManager(mesh, n_dof_per_node=2)

    # BCs: fix left edge (i=0 column, node IDs: j*(n+1)+1 for j=0..n)
    bcs = []
    for j in range(n + 1):
        nid = j * (n + 1) + 1
        bcs.append(EssentialBC(nid, "ux", 0.0))
        bcs.append(EssentialBC(nid, "uy", 0.0))

    # Loads: uniform shear on right edge (i=n column)
    # Total F_y = 1; right edge node IDs: j*(n+1)+(n+1) for j=0..n
    # Equivalent nodal forces for uniform traction: end nodes get 1/(2n), middles 1/n
    nodal_loads = []
    for j in range(n + 1):
        nid = j * (n + 1) + (n + 1)
        if j == 0 or j == n:
            fy = 1.0 / (2 * n)
        else:
            fy = 1.0 / n
        nodal_loads.append(NodalLoad(nid, "uy", fy))

    load_case = LoadCase(nodal_loads=nodal_loads)
    prob = StructuralStaticProblem(
        mesh, dof_manager, elements, bcs, load_case, {"m": mat}
    )
    return prob


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cooks_membrane_2x2_reasonable_range():
    """2×2 mesh gives v_C consistent with Simo & Rifai (1990) reference for Q4.

    Standard displacement Q4 (full 2×2 Gauss) reference: v_C ~ 11.8 for 2×2 mesh.
    Note: the 23.97 converged value requires very fine meshes or enhanced elements.
    """
    prob = build_cooks_membrane(2)
    result = prob.solve()
    # Top-right corner node ID = (n+1)^2 = 9
    v_C = result.nodal_displacements[9]["uy"]
    # Standard Q4 2×2 reference: ~11.8 (Simo & Rifai 1990, Table 2)
    np.testing.assert_allclose(v_C, 11.8, rtol=0.05,
                                err_msg=f"v_C={v_C:.4f} not within 5% of reference 11.8")


def test_cooks_membrane_4x4_converges_toward_reference():
    """4×4 mesh converges toward the reference solution for standard Q4.

    Standard Q4 reference (Simo & Rifai 1990): v_C ~ 18.3 for 4×4 mesh.
    """
    prob = build_cooks_membrane(4)
    result = prob.solve()
    # Top-right corner node ID = 25
    v_C = result.nodal_displacements[25]["uy"]
    # Standard Q4 4×4 reference: ~18.3
    np.testing.assert_allclose(v_C, 18.3, rtol=0.05,
                                err_msg=f"v_C={v_C:.4f} not within 5% of reference 18.3")


def test_cooks_membrane_monotone_convergence():
    """v_C should increase (lower bound convergence) with mesh refinement."""
    v_2 = build_cooks_membrane(2).solve().nodal_displacements[9]["uy"]
    v_4 = build_cooks_membrane(4).solve().nodal_displacements[25]["uy"]
    assert v_2 < v_4, f"Expected v(2×2)={v_2:.4f} < v(4×4)={v_4:.4f}"


# ---------------------------------------------------------------------------
# Simple patch test: single quad under uniform axial stress
# ---------------------------------------------------------------------------

def test_single_quad_uniaxial_tension():
    """Single quad4 (ν=0) under uniform tension: u_tip = P/(E·t), \sigma_xx = P/A."""
    E, nu, t = 200e9, 0.0, 0.01   # ν=0 eliminates Poisson coupling
    L = 1.0
    P = 1000.0
    mat = LinearElasticMaterial("steel", E=E, nu=nu, rho=7850.0, thickness=t)
    # Square element (0,0)→(L,0)→(L,L)→(0,L)
    coords = np.array([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]])
    nodes = [
        Node(1, np.array([0.0, 0.0])),
        Node(2, np.array([L, 0.0])),
        Node(3, np.array([L, L])),
        Node(4, np.array([0.0, L])),
    ]
    mesh = Mesh(nodes=nodes, elements=[Element(1, "quad4", [1, 2, 3, 4], "steel")])
    dof_manager = DOFManager(mesh, n_dof_per_node=2)
    elem = Quad4Element(1, [1, 2, 3, 4], coords, mat, plane="stress")

    bcs = [
        EssentialBC(1, "ux", 0.0), EssentialBC(1, "uy", 0.0),
        EssentialBC(4, "ux", 0.0), EssentialBC(4, "uy", 0.0),
    ]
    loads = LoadCase(nodal_loads=[
        NodalLoad(2, "ux", P / 2.0),   # distribute P over 2 right-edge nodes
        NodalLoad(3, "ux", P / 2.0),
    ])
    prob = StructuralStaticProblem(mesh, dof_manager, [elem], bcs, loads, {"steel": mat})
    result = prob.solve()

    # Average ux on right edge ~ P·L / (E·t·L) = P / (E·t)
    u2x = result.nodal_displacements[2]["ux"]
    u3x = result.nodal_displacements[3]["ux"]
    u_avg = 0.5 * (u2x + u3x)
    expected = P / (E * t)  # PL / (E * A) with A = t*L = L*t for a unit-width problem
    np.testing.assert_allclose(u_avg, expected, rtol=0.01)


# ---------------------------------------------------------------------------
# Tri3 — simple constant-stress patch test
# ---------------------------------------------------------------------------

def test_tri3_patch_constant_strain():
    """Two CST triangles tiling a unit square under uniform tension."""
    E, nu = 1.0, 0.0
    mat = LinearElasticMaterial("m", E=E, nu=nu, rho=0.0, thickness=1.0)
    # Square divided into 2 triangles: (1,2,3) + (1,3,4)
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
            Element(1, "tri3", [1, 2, 3], "m"),
            Element(2, "tri3", [1, 3, 4], "m"),
        ],
    )
    dof_manager = DOFManager(mesh, n_dof_per_node=2)
    elem1 = Tri3Element(1, [1, 2, 3],
                        np.array([[0, 0], [1, 0], [1, 1]], dtype=float), mat)
    elem2 = Tri3Element(2, [1, 3, 4],
                        np.array([[0, 0], [1, 1], [0, 1]], dtype=float), mat)

    eps = 0.001
    # Fix left edge (ux=0) and roller on node 1 (uy=0)
    bcs = [
        EssentialBC(1, "ux", 0.0), EssentialBC(1, "uy", 0.0),
        EssentialBC(4, "ux", 0.0),
    ]
    # Apply consistent nodal forces for \sigma_xx=E*eps: F = \sigma_xx * t * h * Ni
    # For two right-edge nodes (2, 3) sharing height 1: each gets E*eps*t*1/2
    F_node = E * eps * 1.0 * 0.5
    loads = LoadCase(nodal_loads=[
        NodalLoad(2, "ux", F_node),
        NodalLoad(3, "ux", F_node),
    ])
    prob = StructuralStaticProblem(mesh, dof_manager, [elem1, elem2], bcs, loads, {"m": mat})
    result = prob.solve()

    # Both elements should have \sigma_xx ~ E*eps
    for eid, elem in [(1, elem1), (2, elem2)]:
        dofs = dof_manager.get_element_dofs(elem.node_ids)
        u_e = np.array([result.nodal_displacements[nid][dname]
                        for nid in elem.node_ids for dname in ["ux", "uy"]])
        sigma = elem.compute_stress(u_e)
        np.testing.assert_allclose(sigma[0], E * eps, rtol=1e-6,
                                    err_msg=f"Element {eid}: \sigma_xx wrong")

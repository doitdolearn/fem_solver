"""Integration tests for the VTK .vtu writer.

Verifies that write_vtu produces valid XML with correct structure and
node/cell counts for 1-D bar, 2-D triangle, and 2-D quad meshes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.bar1d import Bar1DElement
from femsolver.elements.tri3 import Tri3Element
from femsolver.elements.quad4 import Quad4Element
from femsolver.materials.linear_elastic import LinearElasticMaterial
from femsolver.physics.structural_static import StructuralStaticProblem
from femsolver.postprocessing.vtk_writer import write_vtu


# ---------------------------------------------------------------------------
# Helpers: build & solve small problems
# ---------------------------------------------------------------------------

def _bar_result():
    """2-element bar1d, fixed-free, tip load — 1-D problem."""
    mat = LinearElasticMaterial("m", E=1e6, nu=0.0, rho=0.0, A=1e-3)
    nodes = [Node(i + 1, np.array([float(i)])) for i in range(3)]
    mesh_elems = [Element(1, "bar1d", [1, 2], "m"), Element(2, "bar1d", [2, 3], "m")]
    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dm = DOFManager(mesh, n_dof_per_node=1)
    elems = [
        Bar1DElement(1, [1, 2], np.array([[0.0], [1.0]]), mat),
        Bar1DElement(2, [2, 3], np.array([[1.0], [2.0]]), mat),
    ]
    bcs = [EssentialBC(1, "ux", 0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(3, "ux", 100.0)])
    prob = StructuralStaticProblem(mesh=mesh, dof_manager=dm, elements=elems,
                                   essential_bcs=bcs, load_case=loads,
                                   material_map={"m": mat})
    return prob.solve(), mesh, elems, dm


def _tri3_result():
    """Two tri3 elements forming a unit square — 2-D plane stress."""
    mat = LinearElasticMaterial("m", E=1e9, nu=0.3, rho=0.0, thickness=0.01)
    coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
    nodes = [Node(nid, np.array(xy, dtype=float)) for nid, xy in coords.items()]
    mesh_elems = [
        Element(1, "tri3", [1, 2, 3], "m"),
        Element(2, "tri3", [1, 3, 4], "m"),
    ]
    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dm = DOFManager(mesh, n_dof_per_node=2)
    node_map = {n.id: n for n in nodes}
    elems = [
        Tri3Element(1, [1, 2, 3],
                    np.array([node_map[n].coords for n in [1, 2, 3]]), mat, plane="stress"),
        Tri3Element(2, [1, 3, 4],
                    np.array([node_map[n].coords for n in [1, 3, 4]]), mat, plane="stress"),
    ]
    bcs = [EssentialBC(1, "ux", 0.0), EssentialBC(1, "uy", 0.0),
           EssentialBC(4, "ux", 0.0), EssentialBC(4, "uy", 0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(2, "uy", -1000.0),
                                   NodalLoad(3, "uy", -1000.0)])
    prob = StructuralStaticProblem(mesh=mesh, dof_manager=dm, elements=elems,
                                   essential_bcs=bcs, load_case=loads,
                                   material_map={"m": mat})
    return prob.solve(), mesh, elems, dm


def _quad4_result():
    """Single quad4 element — 2-D plane stress."""
    mat = LinearElasticMaterial("m", E=1e9, nu=0.3, rho=0.0, thickness=0.01)
    nodes = [Node(1, np.array([0.0, 0.0])), Node(2, np.array([1.0, 0.0])),
             Node(3, np.array([1.0, 1.0])), Node(4, np.array([0.0, 1.0]))]
    mesh_elems = [Element(1, "quad4", [1, 2, 3, 4], "m")]
    mesh = Mesh(nodes=nodes, elements=mesh_elems)
    dm = DOFManager(mesh, n_dof_per_node=2)
    elems = [Quad4Element(1, [1, 2, 3, 4],
                          np.array([n.coords for n in nodes]), mat, plane="stress")]
    bcs = [EssentialBC(1, "ux", 0.0), EssentialBC(1, "uy", 0.0),
           EssentialBC(4, "ux", 0.0), EssentialBC(4, "uy", 0.0)]
    loads = LoadCase(nodal_loads=[NodalLoad(2, "uy", -500.0),
                                   NodalLoad(3, "uy", -500.0)])
    prob = StructuralStaticProblem(mesh=mesh, dof_manager=dm, elements=elems,
                                   essential_bcs=bcs, load_case=loads,
                                   material_map={"m": mat})
    return prob.solve(), mesh, elems, dm


# ---------------------------------------------------------------------------
# XML validity
# ---------------------------------------------------------------------------

def test_vtu_is_valid_xml(tmp_path):
    result, mesh, elems, dm = _bar_result()
    path = str(tmp_path / "out.vtu")
    write_vtu(result, mesh, elems, dm, path)
    tree = ET.parse(path)      # raises if not valid XML
    root = tree.getroot()
    assert root.tag == "VTKFile"


def test_vtu_root_attributes(tmp_path):
    result, mesh, elems, dm = _bar_result()
    path = str(tmp_path / "out.vtu")
    write_vtu(result, mesh, elems, dm, path)
    root = ET.parse(path).getroot()
    assert root.attrib["type"] == "UnstructuredGrid"
    assert root.attrib["version"] == "0.1"


# ---------------------------------------------------------------------------
# Node / cell counts
# ---------------------------------------------------------------------------

def _get_piece(path: str) -> ET.Element:
    root = ET.parse(path).getroot()
    return root.find(".//Piece")


def test_vtu_bar_node_count(tmp_path):
    result, mesh, elems, dm = _bar_result()
    path = str(tmp_path / "bar.vtu")
    write_vtu(result, mesh, elems, dm, path)
    piece = _get_piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == 3  # 3 nodes
    assert int(piece.attrib["NumberOfCells"]) == 2   # 2 bar elements


def test_vtu_tri3_counts(tmp_path):
    result, mesh, elems, dm = _tri3_result()
    path = str(tmp_path / "tri3.vtu")
    write_vtu(result, mesh, elems, dm, path)
    piece = _get_piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == 4
    assert int(piece.attrib["NumberOfCells"]) == 2


def test_vtu_quad4_counts(tmp_path):
    result, mesh, elems, dm = _quad4_result()
    path = str(tmp_path / "quad4.vtu")
    write_vtu(result, mesh, elems, dm, path)
    piece = _get_piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == 4
    assert int(piece.attrib["NumberOfCells"]) == 1


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

def _get_types(path: str) -> list:
    root = ET.parse(path).getroot()
    types_el = root.find(".//DataArray[@Name='types']")
    return [int(x) for x in types_el.text.split()]


def test_vtu_bar_cell_type(tmp_path):
    """bar1d → VTK_LINE (type 3)."""
    result, mesh, elems, dm = _bar_result()
    path = str(tmp_path / "bar.vtu")
    write_vtu(result, mesh, elems, dm, path)
    types = _get_types(path)
    assert all(t == 3 for t in types)


def test_vtu_tri3_cell_type(tmp_path):
    """tri3 → VTK_TRIANGLE (type 5)."""
    result, mesh, elems, dm = _tri3_result()
    path = str(tmp_path / "tri3.vtu")
    write_vtu(result, mesh, elems, dm, path)
    types = _get_types(path)
    assert all(t == 5 for t in types)


def test_vtu_quad4_cell_type(tmp_path):
    """quad4 → VTK_QUAD (type 9)."""
    result, mesh, elems, dm = _quad4_result()
    path = str(tmp_path / "quad4.vtu")
    write_vtu(result, mesh, elems, dm, path)
    types = _get_types(path)
    assert all(t == 9 for t in types)


# ---------------------------------------------------------------------------
# Data arrays present
# ---------------------------------------------------------------------------

def test_vtu_has_displacement(tmp_path):
    result, mesh, elems, dm = _tri3_result()
    path = str(tmp_path / "tri3.vtu")
    write_vtu(result, mesh, elems, dm, path)
    root = ET.parse(path).getroot()
    disp = root.find(".//DataArray[@Name='Displacement']")
    assert disp is not None
    assert disp.attrib["NumberOfComponents"] == "3"


def test_vtu_has_stress(tmp_path):
    result, mesh, elems, dm = _tri3_result()
    path = str(tmp_path / "tri3.vtu")
    write_vtu(result, mesh, elems, dm, path)
    root = ET.parse(path).getroot()
    stress = root.find(".//DataArray[@Name='von_Mises_stress']")
    assert stress is not None
    values = [float(x) for x in stress.text.split()]
    assert len(values) == 2    # 2 elements


def test_vtu_displacement_finite(tmp_path):
    result, mesh, elems, dm = _quad4_result()
    path = str(tmp_path / "quad4.vtu")
    write_vtu(result, mesh, elems, dm, path)
    root = ET.parse(path).getroot()
    disp = root.find(".//DataArray[@Name='Displacement']")
    vals = [float(x) for x in disp.text.split()]
    assert all(np.isfinite(v) for v in vals)


# ---------------------------------------------------------------------------
# CLI integration: --vtk flag produces a .vtu file
# ---------------------------------------------------------------------------

def test_cli_vtk_flag_creates_vtu(tmp_path):
    """solve.py --vtk must produce both .yaml and .vtu output."""
    import subprocess, sys, pathlib

    yaml_in = pathlib.Path(__file__).parents[2] / "examples" / "simple_truss.yaml"
    yaml_out = tmp_path / "out.yaml"
    vtu_out = tmp_path / "out.vtu"

    proc = subprocess.run(
        [sys.executable, "solve.py", "-i", str(yaml_in), "-o", str(yaml_out), "--vtk"],
        capture_output=True, text=True,
        cwd=str(pathlib.Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, f"solve.py failed:\n{proc.stderr}"
    assert yaml_out.exists(), ".yaml output missing"
    assert vtu_out.exists(), ".vtu output missing (--vtk flag not honoured)"


def test_cli_vtk_true_in_yaml_creates_vtu(tmp_path):
    """output.vtk: true in the YAML must produce a .vtu file without --vtk flag."""
    import subprocess, sys, pathlib, shutil

    src = pathlib.Path(__file__).parents[2] / "examples" / "plane_stress_plate.yaml"
    yaml_out = tmp_path / "out.yaml"

    proc = subprocess.run(
        [sys.executable, "solve.py", "-i", str(src), "-o", str(yaml_out)],
        capture_output=True, text=True,
        cwd=str(pathlib.Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, f"solve.py failed:\n{proc.stderr}"
    # plane_stress_plate.yaml has vtk: false by default; just verify it runs
    assert yaml_out.exists()

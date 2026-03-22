"""Write SolveResult to VTK XML Unstructured Grid format (.vtu).

The file can be opened directly in ParaView or VisIt.  No external VTK
library is required — the format is generated as plain XML text.

VTK cell type codes used
------------------------
  bar1d, bar2d, bar1d_nl, rod1d_thermal  →   3  (VTK_LINE,     2 nodes)
  tri3,  tri3_thermal                    →   5  (VTK_TRIANGLE, 3 nodes)
  quad4, quad4_axisym                       →   9  (VTK_QUAD,     4 nodes)

All node coordinates and displacement vectors are output as 3-component
(x, y, z) tuples; 1-D problems get y=z=0, 2-D problems get z=0.

Point data
----------
  Displacement  (3 components: ux, uy, 0)

Cell data
---------
  von_Mises_stress  (scalar)
    • 1-D bar: |σ_axial|
    • 2-D continuum: √(σxx² − σxx·σyy + σyy² + 3·τxy²)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np

from femsolver.core.mesh import Mesh
from femsolver.core.dof_manager import DOFManager
from femsolver.physics.base_problem import SolveResult
from femsolver.postprocessing.stress_recovery import von_mises

# Map element type string → VTK cell type integer
_VTK_TYPE: Dict[str, int] = {
    "bar1d":         3,
    "bar2d":         3,
    "bar1d_nl":      3,
    "rod1d_thermal": 3,
    "tri3":          5,
    "tri3_thermal":  5,
    "quad4":         9,
    "quad4_axisym":  9,  # VTK_QUAD (displayed in r-z plane)
}
_VTK_TYPE_DEFAULT = 3   # fall-back: VTK_LINE

def _pad3(coords) -> tuple:
    """Pad a coordinate array to 3 components."""
    x = float(coords[0]) if len(coords) > 0 else 0.0
    y = float(coords[1]) if len(coords) > 1 else 0.0
    z = float(coords[2]) if len(coords) > 2 else 0.0
    return x, y, z


def write_vtu(
    result: SolveResult,
    mesh: Mesh,
    elements: list,
    dof_manager: DOFManager,
    path: str,
) -> None:
    """Write *result* to *path* as a VTK XML Unstructured Grid (.vtu) file.

    Parameters
    ----------
    result      : solved SolveResult (must contain nodal_displacements).
    mesh        : Mesh with Node and Element objects.
    elements    : list of BaseElement objects (same order as mesh.elements).
    dof_manager : used to look up DOF names.
    path        : output file path (should end in ``.vtu``).
    """
    # ------------------------------------------------------------------ #
    # 1. Build node-id → 0-based VTK index map                           #
    # ------------------------------------------------------------------ #
    node_ids = [n.id for n in mesh.nodes]
    nid_to_idx: Dict[int, int] = {nid: i for i, nid in enumerate(node_ids)}
    n_pts = len(node_ids)
    n_cells = len(elements)

    # ------------------------------------------------------------------ #
    # 2. Detect cylindrical elements                                      #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 3. Points (3-D) — convert cylindrical (r,θ,z) → Cartesian (x,y,z) #
    # ------------------------------------------------------------------ #
    points_xyz = [_pad3(n.coords) for n in mesh.nodes]

    # ------------------------------------------------------------------ #
    # 3. Cells (connectivity, offsets, types)                             #
    # ------------------------------------------------------------------ #
    connectivity_flat: List[int] = []
    offsets: List[int] = []
    cell_types: List[int] = []
    offset = 0

    for mesh_elem in mesh.elements:
        vtk_t = _VTK_TYPE.get(mesh_elem.type, _VTK_TYPE_DEFAULT)
        cell_types.append(vtk_t)
        for nid in mesh_elem.node_ids:
            connectivity_flat.append(nid_to_idx[nid])
        offset += len(mesh_elem.node_ids)
        offsets.append(offset)

    # ------------------------------------------------------------------ #
    # 4. Point data: displacement                                         #
    # ------------------------------------------------------------------ #
    dof_names = dof_manager.dof_names
    disp_data: List[tuple] = []

    for nid in node_ids:
        if nid in result.nodal_displacements:
            d = result.nodal_displacements[nid]
            ux = d.get(dof_names[0], 0.0)
            uy = d.get(dof_names[1], 0.0) if len(dof_names) > 1 else 0.0
        else:
            ux, uy = 0.0, 0.0
        disp_data.append((ux, uy, 0.0))

    # ------------------------------------------------------------------ #
    # 5. Cell data: von Mises stress (and pressure if available)          #
    # ------------------------------------------------------------------ #
    vm_data: List[float] = []
    for elem in elements:
        sigma = result.element_stresses.get(elem.element_id, np.zeros(1))
        vm_data.append(von_mises(sigma))

    # ------------------------------------------------------------------ #
    # 6. Build XML tree                                                   #
    # ------------------------------------------------------------------ #
    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid, "Piece",
        NumberOfPoints=str(n_pts),
        NumberOfCells=str(n_cells),
    )

    # Points
    pts_parent = ET.SubElement(piece, "Points")
    _data_array(pts_parent, type="Float64", NumberOfComponents="3",
                text="\n".join(f"{x:.10e} {y:.10e} {z:.10e}" for x, y, z in points_xyz))

    # Cells
    cells_parent = ET.SubElement(piece, "Cells")
    _data_array(cells_parent, type="Int32", Name="connectivity",
                text=" ".join(str(i) for i in connectivity_flat))
    _data_array(cells_parent, type="Int32", Name="offsets",
                text=" ".join(str(o) for o in offsets))
    _data_array(cells_parent, type="UInt8", Name="types",
                text=" ".join(str(t) for t in cell_types))

    # Point data
    point_data = ET.SubElement(piece, "PointData", Vectors="Displacement")
    _data_array(point_data, type="Float64", Name="Displacement",
                NumberOfComponents="3",
                text="\n".join(f"{ux:.10e} {uy:.10e} {uz:.10e}" for ux, uy, uz in disp_data))

    # Cell data
    cell_data = ET.SubElement(piece, "CellData", Scalars="von_Mises_stress")
    _data_array(cell_data, type="Float64", Name="von_Mises_stress",
                text=" ".join(f"{v:.10e}" for v in vm_data))

    # ------------------------------------------------------------------ #
    # 7. Write file                                                       #
    # ------------------------------------------------------------------ #
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_array(parent: ET.Element, *, text: str, **attribs) -> ET.Element:
    """Append a <DataArray> child with ``format="ascii"``."""
    attribs.setdefault("format", "ascii")
    el = ET.SubElement(parent, "DataArray", **attribs)
    el.text = "\n" + text + "\n"
    return el

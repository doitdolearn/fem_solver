from __future__ import annotations

from typing import List, Optional

def generate_axisymmetric_warpage_mesh(
    R: float,
    h_s: float,
    h_f: float,
    n_r: int,
    n_z_substrate: int,
    n_z_film: int,
    r_min: Optional[float] = None,
    mat_id_substrate: str = "substrate",
    mat_id_film: str = "film",
) -> dict:
    """Generate a 2D axisymmetric quad4 mesh in the (r, z) plane.

    Much more efficient than the 3D hex8 mesh for axisymmetric problems.
    Handles extreme thin-film aspect ratios correctly.

    Parameters
    ----------
    R : outer radius
    h_s : substrate thickness
    h_f : film thickness
    n_r : number of radial divisions
    n_z_substrate : number of z divisions in substrate
    n_z_film : number of z divisions in film
    r_min : inner radius (default R/100)
    mat_id_substrate : material id for substrate elements
    mat_id_film : material id for film elements

    Returns
    -------
    dict with keys: nodes, elements, boundary_conditions (YAML-compatible)
    """
    if r_min is None:
        r_min = R / 100.0

    n_z_total = n_z_substrate + n_z_film
    nr1 = n_r + 1
    nz1 = n_z_total + 1

    # --- Node generation ---
    r_vals = [r_min + (R - r_min) * ir / n_r for ir in range(nr1)]
    z_vals: List[float] = []
    for iz in range(n_z_substrate + 1):
        z_vals.append(h_s * iz / n_z_substrate)
    for iz in range(1, n_z_film + 1):
        z_vals.append(h_s + h_f * iz / n_z_film)

    def node_id(ir: int, iz: int) -> int:
        return iz * nr1 + ir + 1  # 1-based

    nodes: List[dict] = []
    for iz in range(nz1):
        for ir in range(nr1):
            nid = node_id(ir, iz)
            nodes.append({
                "id": nid,
                "coords": [r_vals[ir], z_vals[iz]],
            })

    # --- Element generation ---
    elements: List[dict] = []
    elem_id = 1
    for iz in range(n_z_total):
        mat = mat_id_substrate if iz < n_z_substrate else mat_id_film
        for ir in range(n_r):
            # CCW node ordering: bottom-left, bottom-right, top-right, top-left
            n0 = node_id(ir, iz)
            n1 = node_id(ir + 1, iz)
            n2 = node_id(ir + 1, iz + 1)
            n3 = node_id(ir, iz + 1)
            elements.append({
                "id": elem_id,
                "type": "quad4_axisym",
                "nodes": [n0, n1, n2, n3],
                "material": mat,
            })
            elem_id += 1

    # --- Boundary conditions ---
    # Inner radius (r=r_min): fix ur (symmetry)
    # Only one node (bottom inner corner): fix uz (prevent rigid body translation)
    essential_bcs: List[dict] = []
    for iz in range(nz1):
        nid = node_id(0, iz)  # ir=0 -> r=r_min
        if iz == 0:
            # Bottom inner corner: fix both ur and uz
            essential_bcs.append({
                "node": nid,
                "dofs": ["ur", "uz"],
                "values": [0.0, 0.0],
            })
        else:
            # Other z-levels at inner radius: fix only ur
            essential_bcs.append({
                "node": nid,
                "dofs": ["ur"],
                "values": [0.0],
            })

    return {
        "nodes": nodes,
        "elements": elements,
        "boundary_conditions": {"essential": essential_bcs},
    }
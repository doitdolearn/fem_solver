from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from femsolver.core.boundary_conditions import EssentialBC, LoadCase, NaturalBC, NodalLoad
from femsolver.core.dof_manager import DOFManager
from femsolver.core.mesh import Element, Mesh, Node
from femsolver.elements.base import BaseElement
from femsolver.elements.registry import get_element_class
from femsolver.materials.linear_elastic import LinearElasticMaterial
from femsolver.materials.thermal_isotropic import ThermalIsotropicMaterial

# Maps force DOF names (used in YAML loads) to displacement/temperature DOF names
_FORCE_TO_DISP: Dict[str, str] = {
    "fx": "ux",
    "fy": "uy",
    "fz": "uz",
    "fr": "ur",
    "ftheta": "utheta",
    "Q":  "T",    # thermal: nodal heat source → temperature DOF
}


def build_model(
    data: dict,
) -> Tuple[Mesh, DOFManager, List[BaseElement], Dict, List[EssentialBC], LoadCase, dict]:
    """
    Convert a validated input dict into domain objects.

    Returns:
        (mesh, dof_manager, elements, material_map, essential_bcs, load_case, problem_config)
    """
    problem_config = data["problem"]
    problem_type: str = problem_config.get("type", "structural_static")
    dimension: int = problem_config.get("dimension", 1)
    plane_condition: str = problem_config.get("plane_condition", None)
    coordinate_system: str = problem_config.get("coordinate_system", "cartesian")

    # Thermal load (optional top-level key)
    thermal_load = data.get("thermal_load", {})
    delta_T: float = float(thermal_load.get("delta_T", 0.0)) if thermal_load else 0.0

    # Map YAML plane_condition → element-level plane keyword
    _plane_map = {"plane_stress": "stress", "plane_strain": "strain"}
    plane = _plane_map.get(plane_condition, "stress")

    # --- Materials ---
    material_map: Dict = {}
    for mat_data in data["materials"]:
        mat = _build_material(mat_data)
        material_map[mat.id] = mat

    # --- Mesh ---
    nodes = [
        Node(id=n["id"], coords=np.array(n["coords"], dtype=float))
        for n in data["nodes"]
    ]
    mesh_elements = [
        Element(
            id=e["id"],
            type=e["type"],
            node_ids=e["nodes"],
            material_id=e["material"],
        )
        for e in data["elements"]
    ]
    mesh = Mesh(nodes=nodes, elements=mesh_elements)

    # --- DOF manager ---
    # Thermal problems always have 1 DOF per node (temperature "T")
    if problem_type == "thermal_steady":
        dof_manager = DOFManager(mesh, n_dof_per_node=1, dof_names=["T"])
    elif coordinate_system == "cylindrical":
        dof_manager = DOFManager(mesh, n_dof_per_node=3, dof_names=["ur", "utheta", "uz"])
    elif coordinate_system == "axisymmetric":
        dof_manager = DOFManager(mesh, n_dof_per_node=2, dof_names=["ur", "uz"])
    else:
        dof_manager = DOFManager(mesh, n_dof_per_node=dimension)

    # --- Computational element objects ---
    node_map: Dict[int, Node] = {n.id: n for n in nodes}
    elements: List[BaseElement] = []
    for e_data in data["elements"]:
        elem_class = get_element_class(e_data["type"])
        node_ids = e_data["nodes"]
        coords = np.array([node_map[nid].coords for nid in node_ids])
        mat = material_map[e_data["material"]]
        elem = elem_class(
            element_id=e_data["id"],
            node_ids=node_ids,
            nodes_coords=coords,
            material=mat,
            plane=plane,
            delta_T=delta_T,
        )
        elements.append(elem)

    # --- Essential BCs ---
    essential_bcs: List[EssentialBC] = []
    for bc in data["boundary_conditions"].get("essential", []) or []:
        node_id = bc["node"]
        for dof_name, value in zip(bc["dofs"], bc["values"]):
            essential_bcs.append(
                EssentialBC(node_id=node_id, dof_name=dof_name, value=float(value))
            )

    # --- Load case ---
    nodal_loads: List[NodalLoad] = []
    for load in data["loads"].get("nodal", []) or []:
        node_id = load["node"]
        for i, dof_name in enumerate(load["dofs"]):
            # Convert force/flux name (fx, Q) → displacement/temperature DOF name (ux, T)
            disp_name = _FORCE_TO_DISP.get(dof_name, dof_name)
            if "values" in load:
                value = float(load["values"][i])
                nodal_loads.append(
                    NodalLoad(node_id=node_id, dof_name=disp_name, value=value)
                )
            else:
                raise ValueError(f"Invalid nodal load {disp_name}")

    load_case = LoadCase(nodal_loads=nodal_loads)

    return mesh, dof_manager, elements, material_map, essential_bcs, load_case, problem_config


def _build_material(mat_data: dict):
    mat_type = mat_data.get("type", "linear_elastic")
    if mat_type == "linear_elastic":
        return LinearElasticMaterial(
            material_id=mat_data["id"],
            E=float(mat_data["E"]),
            nu=float(mat_data["nu"]),
            rho=float(mat_data.get("rho", 0.0)),
            A=float(mat_data["A"]) if "A" in mat_data else None,
            thickness=float(mat_data["thickness"]) if "thickness" in mat_data else None,
            alpha=float(mat_data.get("alpha", 0.0)),
        )
    if mat_type == "thermal_isotropic":
        return ThermalIsotropicMaterial(
            material_id=mat_data["id"],
            k_cond=float(mat_data["k_cond"]),
            cp=float(mat_data.get("cp", 0.0)),
            rho=float(mat_data.get("rho", 0.0)),
            A=float(mat_data["A"]) if "A" in mat_data else None,
            thickness=float(mat_data["thickness"]) if "thickness" in mat_data else None,
        )
    raise ValueError(
        f"Unknown material type: '{mat_type}'. "
        f"Currently supported: 'linear_elastic', 'thermal_isotropic'"
    )

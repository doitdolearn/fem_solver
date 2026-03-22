from __future__ import annotations

_VALID_PROBLEM_TYPES = {
    "structural_static",
    "thermal_steady",
    "modal",
    "nonlinear_static",
    "structural_dynamic",
    "cylindrical_warpage",
}

_VALID_PLANE_CONDITIONS = {"plane_stress", "plane_strain", "none"}
_VALID_FORMULATIONS = {"mixed_up"}


def validate(data: dict) -> None:
    """Validate the top-level structure of an input YAML dict.

    Raises ValueError with a descriptive message on the first error found.
    """
    _check_top_level_keys(data)
    _check_problem(data["problem"])
    _check_nodes(data["nodes"])
    _check_elements(data["elements"])
    _check_boundary_conditions(data["boundary_conditions"])
    _check_loads(data["loads"])
    _check_solver(data["solver"])


_VALID_COORDINATE_SYSTEMS = {"cartesian", "cylindrical", "axisymmetric"}


def _check_top_level_keys(data: dict) -> None:
    required = ["problem", "materials", "nodes", "elements",
                "boundary_conditions", "loads", "solver"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required top-level key: '{key}'")


def _check_problem(problem: dict) -> None:
    for key in ["name", "type"]:
        if key not in problem:
            raise ValueError(f"Missing 'problem.{key}'")

    ptype = problem["type"]
    if ptype not in _VALID_PROBLEM_TYPES:
        raise ValueError(
            f"Unknown problem.type: '{ptype}'. "
            f"Valid types: {sorted(_VALID_PROBLEM_TYPES)}"
        )

    plane = problem.get("plane_condition")
    if plane is not None and plane not in _VALID_PLANE_CONDITIONS:
        raise ValueError(
            f"Unknown problem.plane_condition: '{plane}'. "
            f"Valid values: {sorted(_VALID_PLANE_CONDITIONS)} "
            f"(omit this field for 1D bar problems)"
        )

    formulation = problem.get("formulation")
    if formulation is not None and formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"Unknown problem.formulation: '{formulation}'. "
            f"Valid values: {sorted(_VALID_FORMULATIONS)}"
        )

    coord_sys = problem.get("coordinate_system")
    if coord_sys is not None and coord_sys not in _VALID_COORDINATE_SYSTEMS:
        raise ValueError(
            f"Unknown problem.coordinate_system: '{coord_sys}'. "
            f"Valid values: {sorted(_VALID_COORDINATE_SYSTEMS)}"
        )


def _check_nodes(nodes) -> None:
    if not nodes:
        raise ValueError("At least one node is required")
    for node in nodes:
        for key in ["id", "coords"]:
            if key not in node:
                raise ValueError(f"Node missing '{key}': {node}")


def _check_elements(elements) -> None:
    if not elements:
        raise ValueError("At least one element is required")
    for elem in elements:
        for key in ["id", "type", "nodes", "material"]:
            if key not in elem:
                raise ValueError(f"Element missing '{key}': {elem}")


def _check_boundary_conditions(bcs: dict) -> None:
    for bc in bcs.get("essential", []) or []:
        for key in ["node", "dofs", "values"]:
            if key not in bc:
                raise ValueError(f"Essential BC missing '{key}': {bc}")
        if len(bc["dofs"]) != len(bc["values"]):
            raise ValueError(
                f"Essential BC 'dofs' and 'values' must have the same length: {bc}"
            )


def _check_loads(loads: dict) -> None:
    if not loads:
        return
    for load in loads.get("nodal", []) or []:
        for key in ["node", "dofs"]:
            if key not in load:
                raise ValueError(f"Nodal load missing '{key}': {load}")
        has_values = "values" in load
        has_dynamic = "amplitude" in load and "time_function" in load
        if not has_values and not has_dynamic:
            raise ValueError(
                f"Nodal load must have 'values' or both 'amplitude' and "
                f"'time_function': {load}"
            )


def _check_solver(solver: dict) -> None:
    if "type" not in solver:
        raise ValueError("Missing 'solver.type'")

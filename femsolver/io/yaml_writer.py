from __future__ import annotations

import os
from typing import Optional, List

import yaml

from femsolver.physics.base_problem import SolveResult
from femsolver.core.mesh import Mesh
from femsolver.elements.base import BaseElement
from femsolver.core.dof_manager import DOFManager

_THERMAL = "thermal_steady"


def write_yaml(
    result: SolveResult,
    path: str,
    output_config: dict = None,
    *,
    mesh: Optional[Mesh]=None,
    elements: Optional[List[BaseElement]]=None,
    dof_manager: Optional[DOFManager]=None,
) -> None:
    """Write a SolveResult to a YAML output file.

    When ``output_config["vtk"] is True`` **and** *mesh*, *elements*, and
    *dof_manager* are provided, a ``.vtu`` file is also written alongside the
    YAML output (same stem, different extension).
    """
    cfg = output_config or {}
    is_thermal = result.problem_type == _THERMAL

    out: dict = {
        "metadata": {
            "solver_version": "0.1.0",
            "problem_type": result.problem_type,
            "problem_name": result.problem_name,
            "solve_time_s": round(result.solve_time_s, 6),
            "n_dofs": result.n_dofs,
            "converged": result.converged,
        }
    }

    # Nodal DOF results — temperatures (thermal) or displacements (structural)
    if cfg.get("nodal_displacements", True) and result.nodal_displacements:
        dof_names = list(next(iter(result.nodal_displacements.values())).keys())
        if is_thermal:
            section_key = "nodal_temperatures"
            units = "K"
        else:
            section_key = "nodal_displacements"
            units = "m"
        out[section_key] = {
            "units": units,
            "fields": dof_names,
            "data": [
                [node_id] + [result.nodal_displacements[node_id][name] for name in dof_names]
                for node_id in sorted(result.nodal_displacements.keys())
            ],
        }

    # Element results — heat fluxes (thermal) or stresses (structural)
    if cfg.get("element_stresses", True) and result.element_stresses:
        if is_thermal:
            section_key = "element_heat_fluxes"
            units = "W/m2"
        else:
            section_key = "element_stresses"
            units = "Pa"
        out[section_key] = {
            "units": units,
            "data": [
                [eid] + result.element_stresses[eid].tolist()
                for eid in sorted(result.element_stresses.keys())
            ],
        }

    # Reaction results — heat flows (thermal) or forces (structural)
    if cfg.get("reaction_forces", True) and result.reaction_forces:
        all_names: set = set()
        for vals in result.reaction_forces.values():
            all_names.update(vals.keys())
        names = sorted(all_names)
        if is_thermal:
            section_key = "heat_flow_reactions"
            units = "W"
        else:
            section_key = "reaction_forces"
            units = "N"
        out[section_key] = {
            "units": units,
            "fields": names,
            "data": [
                [node_id]
                + [result.reaction_forces[node_id].get(name, 0.0) for name in names]
                for node_id in sorted(result.reaction_forces.keys())
            ],
        }

    if cfg.get("element_pressures", False) and result.element_pressures:
        out["element_pressures"] = {
            "units": "Pa",
            "data": [
                [eid, result.element_pressures[eid]]
                for eid in sorted(result.element_pressures.keys())
            ],
        }

    # Structural dynamics: time-history section
    if result.problem_type == "structural_dynamic" and result.time_history:
        dof_names = list(next(iter(result.nodal_displacements.values())).keys())
        steps_out = []
        for step in result.time_history:
            steps_out.append({
                "time": step["time"],
                "data": [
                    [nid] + [step["nodal_displacements"][nid][name] for name in dof_names]
                    for nid in sorted(step["nodal_displacements"].keys())
                ],
            })
        out["time_history"] = {
            "units": "m",
            "fields": dof_names,
            "n_steps": len(steps_out),
            "steps": steps_out,
        }

    # Modal analysis output
    if result.problem_type == "modal" and result.natural_frequencies_hz:
        modes_out = []
        for i, (f_hz, f_rad) in enumerate(
            zip(result.natural_frequencies_hz, result.natural_frequencies_rad)
        ):
            mode_data = {
                "mode": i + 1,
                "frequency_hz": round(f_hz, 4),
                "frequency_rad_s": round(f_rad, 4),
            }
            if i < len(result.mode_shapes) and result.mode_shapes[i]:
                dof_names = list(next(iter(result.mode_shapes[i].values())).keys())
                mode_data["shape"] = {
                    "fields": dof_names,
                    "data": [
                        [nid] + [result.mode_shapes[i][nid][name] for name in dof_names]
                        for nid in sorted(result.mode_shapes[i])
                    ],
                }
            modes_out.append(mode_data)
        out["modal_results"] = {"n_modes": len(modes_out), "modes": modes_out}

    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    # VTK output (triggered by output.vtk: true or --vtk CLI flag)
    # if cfg.get("vtk", False) and mesh is not None and elements is not None and dof_manager is not None:
    #     from femsolver.postprocessing.vtk_writer import write_vtu
    #     stem = os.path.splitext(path)[0]
    #     vtu_path = stem + ".vtu"
    #     write_vtu(result, mesh, elements, dof_manager, vtu_path)

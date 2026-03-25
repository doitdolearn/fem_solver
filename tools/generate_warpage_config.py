#!/usr/bin/env python
"""Generate a YAML input file for a cylindrical warpage (thin film on substrate) problem.

Usage example:
    python tools/generate_warpage_config.py \
        --h_f 1e-6 --h_s 500e-6 --R 1e-1 \
        --E_f 200e9 --E_s 130e9 --nu_f 0.3 --nu_s 0.28 \
        --alpha_f 14e-6 --alpha_s 2.6e-6 \
        --temp_init 400 --temp_final 25 \
        -o warpage_input.yaml

Formulations:
    (default)         3D cylindrical hex8_cyl (r, theta, z coords)
    --axisymmetric    2D axisymmetric quad4 (r, z coords) — fastest & most accurate
"""
import argparse
import sys
import os

import yaml

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femsolver.utils.mesh_generator.warpage_mesh_generator import (
    generate_axisymmetric_warpage_mesh,
)


def main():
    p = argparse.ArgumentParser(description="Generate cylindrical warpage YAML config")
    # Geometry
    p.add_argument("--h_f", type=float, required=True, help="Film thickness [m]")
    p.add_argument("--h_s", type=float, required=True, help="Substrate thickness [m]")
    p.add_argument("--R", type=float, required=True, help="Disk radius [m]")
    p.add_argument("--r_min", type=float, default=None,
                   help="Inner radius [m] (default: R/10000 for axisym/cart3d, R/100 for cyl3d)")
    # Film material
    p.add_argument("--E_f", type=float, required=True, help="Film Young's modulus [Pa]")
    p.add_argument("--nu_f", type=float, required=True, help="Film Poisson's ratio")
    p.add_argument("--alpha_f", type=float, required=True, help="Film CTE [1/K]")
    p.add_argument("--rho_f", type=float, default=8000.0, help="Film density [kg/m3]")
    # Substrate material
    p.add_argument("--E_s", type=float, required=True, help="Substrate Young's modulus [Pa]")
    p.add_argument("--nu_s", type=float, required=True, help="Substrate Poisson's ratio")
    p.add_argument("--alpha_s", type=float, required=True, help="Substrate CTE [1/K]")
    p.add_argument("--rho_s", type=float, default=2330.0, help="Substrate density [kg/m3]")
    # Temperatures
    p.add_argument("--temp_init", type=float, required=True, help="Initial (stress-free) temperature [K or C]")
    p.add_argument("--temp_final", type=float, required=True, help="Final temperature [K or C]")
    # Mesh density
    p.add_argument("--n_r", type=int, default=None, help="Radial divisions")
    p.add_argument("--n_theta", type=int, default=None, help="Angular divisions (3D only)")
    p.add_argument("--n_z_substrate", type=int, default=None, help="Z divisions in substrate")
    p.add_argument("--n_z_film", type=int, default=None, help="Z divisions in film")
    # Formulation (mutually exclusive)
    form = p.add_mutually_exclusive_group()
    form.add_argument("--axisymmetric", default=True, action="store_true",
                      help="2D axisymmetric quad4 (fastest, most accurate)")
    # Output
    p.add_argument("-o", "--output", required=True, help="Output YAML path")

    args = p.parse_args()

    delta_T = args.temp_final - args.temp_init

    # --- Set mesh defaults based on formulation ---
    n_r = args.n_r or 200
    n_theta = 1  # unused
    n_z_substrate = args.n_z_substrate or 40
    n_z_film = args.n_z_film or 4
    r_min = args.r_min if args.r_min is not None else args.R / 10000.0

    # --- Generate mesh ---
    # if args.axisymmetric:
    mesh_data = generate_axisymmetric_warpage_mesh(
        R=args.R, h_s=args.h_s, h_f=args.h_f,
        n_r=n_r, n_z_substrate=n_z_substrate, n_z_film=n_z_film,
        r_min=r_min,
    )
    coord_system = "axisymmetric"
    dimension = 2

    config = {
        "problem": {
            "name": "cylindrical_warpage_analysis",
            "type": "cylindrical_warpage",
            "dimension": dimension,
            "coordinate_system": coord_system,
        },
        "materials": [
            {
                "id": "substrate",
                "type": "linear_elastic",
                "E": args.E_s,
                "nu": args.nu_s,
                "rho": args.rho_s,
                "alpha": args.alpha_s,
            },
            {
                "id": "film",
                "type": "linear_elastic",
                "E": args.E_f,
                "nu": args.nu_f,
                "rho": args.rho_f,
                "alpha": args.alpha_f,
            },
        ],
        "thermal_load": {
            "delta_T": delta_T,
        },
        "nodes": mesh_data["nodes"],
        "elements": mesh_data["elements"],
        "boundary_conditions": mesh_data["boundary_conditions"],
        "loads": {"nodal": []},
        "solver": {"type": "direct"},
        "output": {
            "nodal_displacements": True,
            "element_stresses": True,
            "reaction_forces": True,
        },
    }

    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    n_nodes = len(mesh_data["nodes"])
    n_elems = len(mesh_data["elements"])
    mode = "axisymmetric"
    print(f"Generated {args.output}: {n_nodes} nodes, {n_elems} elements, "
          f"delta_T={delta_T}, mode={mode}")


if __name__ == "__main__":
    main()

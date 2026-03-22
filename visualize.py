"""Command-line interface for visualizing FemSolverClaude results.

Usage
-----
python visualize.py -i input.yaml -r result.yaml [options]
python visualize.py -i input.yaml --mesh-only [options]

Options
-------
  -i, --input   PATH     Input YAML (mesh geometry)              [required]
  -r, --result  PATH     Result YAML produced by solve.py        [required unless --mesh-only]
  -o, --output  PATH     Save figure to this path (PNG/PDF/SVG …)
      --scale   FLOAT    Displacement magnification (default: 1.0)
      --show             Open interactive plot window
      --node    INT      Node ID for dynamics time-history (default: last)
      --revolve          Revolve axisymmetric result into 3D surface
      --n-theta INT      Angular divisions for 3D revolution (default: 60)
      --mesh-only        Visualize the mesh geometry only (no result file needed)
"""
from __future__ import annotations

import argparse
import sys

import yaml


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Visualize FemSolverClaude input/result YAML files.",
    )
    p.add_argument("-i", "--input",  required=True, metavar="PATH",
                   help="Input YAML (mesh geometry)")
    p.add_argument("-r", "--result", required=False, metavar="PATH", default=None,
                   help="Result YAML from solve.py (not needed with --mesh-only)")
    p.add_argument("-o", "--output", metavar="PATH", default=None,
                   help="Save figure to file")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Displacement magnification factor (default: 1.0)")
    p.add_argument("--show", action="store_true",
                   help="Display interactive plot window")
    p.add_argument("--node", type=int, default=None,
                   help="Node ID for dynamics time-history (default: last node)")
    p.add_argument("--revolve", action="store_true",
                   help="Revolve axisymmetric result into 3D surface plot")
    p.add_argument("--n-theta", type=int, default=60,
                   help="Number of angular divisions for 3D revolution (default: 60)")
    p.add_argument("--mesh-only", action="store_true",
                   help="Visualize mesh geometry only (no result file needed)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Load mesh from input YAML
    from femsolver.io.yaml_reader import load_yaml
    from femsolver.io.model_builder import build_model

    raw = load_yaml(args.input)
    mesh, *_ = build_model(raw)

    # --- Mesh-only mode ---
    if args.mesh_only:
        from femsolver.postprocessing.plot_results import plot_mesh_2d, plot_mesh_3d

        dimension = raw.get("problem", {}).get("dimension", 2)
        coord_sys = raw.get("problem", {}).get("coordinate_system", "cartesian")

        if dimension == 3:
            fig = plot_mesh_3d(mesh, coord_system=coord_sys)
        else:
            fig = plot_mesh_2d(mesh)

    else:
        # --- Result-based visualization ---
        if args.result is None:
            print("Error: -r/--result is required unless --mesh-only is used.",
                  file=sys.stderr)
            sys.exit(1)

        with open(args.result) as f:
            result_data = yaml.safe_load(f)

        problem_type: str = result_data.get("metadata", {}).get("problem_type", "")

        from femsolver.postprocessing.plot_results import (
            plot_structural,
            plot_thermal,
            plot_axisym_3d,
        )

        if args.revolve:
            fig = plot_axisym_3d(mesh, result_data, scale=args.scale,
                                 n_theta=args.n_theta)
        elif problem_type in {"structural_static", "nonlinear_static"}:
            fig = plot_structural(mesh, result_data, scale=args.scale)
        elif problem_type == "thermal_steady":
            fig = plot_thermal(mesh, result_data)
        else:
            print(
                f"Unknown problem type: '{problem_type}'. "
                f"Supported: structural_static, nonlinear_static, thermal_steady",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {args.output}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()

    if not args.output and not args.show:
        print("Tip: use --show to display or -o FILE to save the figure.")


if __name__ == "__main__":
    main()

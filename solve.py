"""FemSolverClaude entry point.

Usage:
    python solve.py -i input.yaml -o output.yaml [--solver direct] [--vtk] [--verbose]
"""

import argparse
import sys
from femsolver.postprocessing.vtk_writer import write_vtu
_PROBLEM_DISPATCH = {
    "structural_static":    "_solve_structural_static",
    "thermal_steady":       "_solve_thermal_steady",
    "nonlinear_static":     "_solve_nonlinear_static",
    "cylindrical_warpage":  "_solve_cylindrical_warpage",
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="femsolver",
        description="Finite Element Method solver",
    )
    parser.add_argument("-i", "--input",  required=True, metavar="PATH", help="Input YAML file")
    parser.add_argument("-o", "--output", required=True, metavar="PATH", help="Output YAML file")
    parser.add_argument(
        "--solver",
        choices=["direct", "cg", "gmres", "eigsh", "newton_raphson", "newmark"],
        default=None,
        help="Override solver type from input file",
    )
    parser.add_argument("--vtk",     action="store_true", help="Write .vtu output for ParaView")
    parser.add_argument("--verbose", default=True, action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    from femsolver.io.yaml_reader import load_yaml
    from femsolver.io.input_validator import validate
    from femsolver.io.model_builder import build_model
    from femsolver.io.yaml_writer import write_yaml

    if args.verbose:
        print(f"[femsolver] Reading {args.input}")

    data = load_yaml(args.input)
    validate(data)

    mesh, dof_manager, elements, material_map, essential_bcs, load_case, problem_config = (
        build_model(data)
    )

    problem_type = problem_config["type"]
    output_config = data.get("output", {})
    if problem_type == "structural_static":
        from femsolver.physics.structural_static import StructuralStaticProblem
        problem = StructuralStaticProblem(
            mesh=mesh,
            dof_manager=dof_manager,
            elements=elements,
            essential_bcs=essential_bcs,
            load_case=load_case,
            material_map=material_map,
            problem_name=problem_config.get("name", "unnamed"),
        )
        result = problem.solve()
    elif problem_type == "thermal_steady":
        from femsolver.physics.thermal_steady import ThermalSteadyProblem
        problem = ThermalSteadyProblem(
            mesh=mesh,
            dof_manager=dof_manager,
            elements=elements,
            essential_bcs=essential_bcs,
            load_case=load_case,
            material_map=material_map,
            problem_name=problem_config.get("name", "unnamed"),
        )
        result = problem.solve()
    elif problem_type == "nonlinear_static":
        from femsolver.physics.nonlinear_static import NonlinearStaticProblem
        solver_cfg = data.get("solver", {})
        n_load_steps = int(solver_cfg.get("n_load_steps", 10))
        nl_tol = float(solver_cfg.get("tolerance", 1e-10))
        max_iter = int(solver_cfg.get("max_iter", 50))
        from femsolver.solvers.newton_raphson import NewtonRaphsonSolver
        problem = NonlinearStaticProblem(
            mesh=mesh,
            dof_manager=dof_manager,
            elements=elements,
            essential_bcs=essential_bcs,
            load_case=load_case,
            material_map=material_map,
            problem_name=problem_config.get("name", "unnamed"),
            n_load_steps=n_load_steps,
            solver=NewtonRaphsonSolver(tol=nl_tol, max_iter=max_iter),
        )
        result = problem.solve()
    elif problem_type == "cylindrical_warpage":
        from femsolver.physics.structural_static import StructuralStaticProblem
        problem = StructuralStaticProblem(
            mesh=mesh,
            dof_manager=dof_manager,
            elements=elements,
            essential_bcs=essential_bcs,
            load_case=load_case,
            material_map=material_map,
            problem_name=problem_config.get("name", "unnamed"),
        )
        result = problem.solve()
    else:
        print(
            f"[femsolver] Problem type '{problem_type}' is not yet implemented. "
            f"Implemented types: {sorted(_PROBLEM_DISPATCH.keys())}"
        )
        sys.exit(1)

    # --vtk CLI flag overrides the YAML output.vtk setting
    if args.vtk:
        output_config = dict(output_config)
        output_config["vtk"] = True

    write_yaml(
        result, args.output, output_config,
        mesh=mesh, elements=elements, dof_manager=dof_manager,
    )

    if args.verbose:
        from femsolver.utils.logger import get_logger
        log = get_logger()
        log.info(
            f"Solved '{result.problem_name}' in {result.solve_time_s:.4f}s "
            f"({result.n_dofs} DOFs) — output: {args.output}"
        )
        if output_config.get("vtk"):
            import os
            vtu_path = os.path.splitext(args.output)[0] + ".vtu"
            log.info(f"VTK output: {vtu_path}")
            write_vtu(result, mesh, elements, dof_manager, vtu_path)


if __name__ == "__main__":
    main()

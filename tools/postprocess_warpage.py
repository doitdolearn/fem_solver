"""
Post-process cylindrical warpage results.

Handles all three formulations:
  - Axisymmetric (ur, uz) with coords (r, z)
  - Cylindrical 3D (ur, utheta, uz) with coords (r, theta, z)
  - Cartesian 3D (ux, uy, uz) with coords (x, y, z)

Usage:
    python tools/postprocess_warpage.py -i input.yaml -r result.yaml
    python tools/postprocess_warpage.py -i input.yaml -r result.yaml --h_s 500e-6 --h_f 1e-6
"""
import argparse
import yaml
import numpy as np


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract warpage metrics from FEM results")
    parser.add_argument("-i", "--input", required=True, help="Input YAML (node coordinates)")
    parser.add_argument("-r", "--result", required=True, help="Result YAML (displacements)")
    parser.add_argument("--h_s", type=float, default=None, help="Substrate thickness [m]")
    parser.add_argument("--h_f", type=float, default=None, help="Film thickness [m]")
    args = parser.parse_args()

    inp = load_yaml(args.input)
    res = load_yaml(args.result)

    # Build node coordinate map
    coords = {n["id"]: np.array(n["coords"]) for n in inp["nodes"]}

    # Build displacement map
    disp_map = {}
    for row in res["nodal_displacements"]["data"]:
        node_id = row[0]
        disp_map[node_id] = np.array(row[1:], dtype=float)

    # Detect coordinate system from DOF fields
    fields = res["nodal_displacements"]["fields"]
    coord_sys = inp.get("problem", {}).get("coordinate_system", "cartesian")

    if coord_sys == "axisymmetric":
        # Coords: (r, z), DOFs: (ur, uz)
        uz_idx = fields.index("uz")
        def get_r(c): return c[0]
        def get_z(c): return c[1]
    elif coord_sys == "cylindrical":
        # Coords: (r, theta, z), DOFs: (ur, utheta, uz)
        uz_idx = fields.index("uz")
        def get_r(c): return c[0]
        def get_z(c): return c[2]
    else:
        # Cartesian: coords (x, y, z), DOFs: (ux, uy, uz)
        uz_idx = fields.index("uz")
        def get_r(c): return np.sqrt(c[0]**2 + c[1]**2)
        def get_z(c): return c[2]

    # Find top z level
    all_z = np.array([get_z(c) for c in coords.values()])
    z_top = all_z.max()
    z_tol = (z_top - all_z.min()) * 1e-6

    # Collect (r, uz) for top-surface nodes
    top_r = []
    top_uz = []
    for node_id, coord in coords.items():
        if abs(get_z(coord) - z_top) < z_tol:
            if node_id in disp_map:
                top_r.append(get_r(coord))
                top_uz.append(disp_map[node_id][uz_idx])

    top_r = np.array(top_r)
    top_uz = np.array(top_uz)

    # Sort by r
    order = np.argsort(top_r)
    top_r = top_r[order]
    top_uz = top_uz[order]

    # Max deflection (relative warpage)
    uz_min = top_uz.min()
    uz_max = top_uz.max()
    w_max = uz_max - uz_min

    # Fit parabola: uz = a*r^2 + b  =>  R_c = 1 / (2*a)
    A = np.column_stack([top_r**2, np.ones_like(top_r)])
    coeffs, _, _, _ = np.linalg.lstsq(A, top_uz, rcond=None)
    a_coeff = coeffs[0]
    R_c = 1.0 / (2.0 * a_coeff) if abs(a_coeff) > 1e-30 else float("inf")

    # --- Stoney analytical reference ---
    has_stoney = False
    try:
        mats = {m["id"]: m for m in inp["materials"]}
        dt = inp["thermal_load"]["delta_T"]
        E_f = mats["film"]["E"]
        nu_f = mats["film"]["nu"]
        alpha_f = mats["film"]["alpha"]
        E_s = mats["substrate"]["E"]
        nu_s = mats["substrate"]["nu"]
        alpha_s = mats["substrate"]["alpha"]
        R_outer = top_r.max()

        # Determine h_s and h_f
        if args.h_s is not None and args.h_f is not None:
            h_s = args.h_s
            h_f = args.h_f
        else:
            # Auto-detect from element z-ranges
            sub_z_max = 0.0
            film_z_min = z_top
            for elem in inp["elements"]:
                elem_z = [get_z(coords[nid]) for nid in elem["nodes"]]
                if elem["material"] == "substrate":
                    sub_z_max = max(sub_z_max, max(elem_z))
                elif elem["material"] == "film":
                    film_z_min = min(film_z_min, min(elem_z))
            z_interface = (sub_z_max + film_z_min) / 2.0
            h_s = z_interface
            h_f = z_top - z_interface

        # Biaxial moduli
        M_f = E_f / (1.0 - nu_f)
        M_s = E_s / (1.0 - nu_s)

        # Film stress from mismatch strain
        eps_mismatch = (alpha_f - alpha_s) * dt
        sigma_f = -M_f * eps_mismatch

        # Stoney formula
        R_c_stoney = M_s * h_s**2 / (6.0 * sigma_f * h_f)
        w_stoney = R_outer**2 / (2.0 * abs(R_c_stoney))
        has_stoney = True
    except Exception:
        pass

    mode_label = {"axisymmetric": "Axisymmetric", "cylindrical": "Cylindrical 3D",
                  "cartesian": "Cartesian 3D"}.get(coord_sys, coord_sys)

    print("=" * 60)
    print(f"  Cylindrical Warpage — FEM Results ({mode_label})")
    print("=" * 60)
    print(f"  Top surface z         = {z_top*1e6:.2f} µm")
    print(f"  Nodes on top surface  = {len(top_r)}")
    print(f"  r range               = [{top_r.min()*1e3:.2f}, {top_r.max()*1e3:.2f}] mm")
    print()
    print(f"  Max uz (absolute)     = {uz_max*1e6:.4f} µm")
    print(f"  Min uz (absolute)     = {uz_min*1e6:.4f} µm")
    print(f"  Relative warpage      = {w_max*1e6:.4f} µm")
    print(f"  Radius of curvature   = {abs(R_c):.2f} m")
    if has_stoney:
        print()
        print("  --- Stoney's Formula (Analytical Reference) ---")
        print(f"  h_s                   = {h_s*1e6:.1f} µm")
        print(f"  h_f                   = {h_f*1e6:.1f} µm")
        print(f"  M_f = E_f/(1-v_f)    = {M_f/1e9:.2f} GPa")
        print(f"  M_s = E_s/(1-v_s)    = {M_s/1e9:.2f} GPa")
        print(f"  Mismatch strain e_m   = {eps_mismatch:.6e}")
        print(f"  Film stress sigma_f   = {sigma_f/1e6:.2f} MPa")
        print(f"  R_c (Stoney)          = {abs(R_c_stoney):.2f} m")
        print(f"  w_max (Stoney)        = {w_stoney*1e6:.4f} µm")
        print()
        print("  --- Comparison ---")
        print(f"  FEM / Stoney (w_max)  = {w_max / w_stoney:.3f}")
        print(f"  FEM / Stoney (R_c)    = {abs(R_c / R_c_stoney):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

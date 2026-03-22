"""Matplotlib-based visualization of FemSolverClaude results.

Public API
----------
plot_structural(mesh, result_data, scale=1.0, ax=None) -> Figure
plot_thermal(mesh, result_data, ax=None)               -> Figure
plot_modal(mesh, result_data, max_modes=6)             -> Figure
plot_dynamic(result_data, node_id=None)                -> Figure
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from femsolver.core.mesh import Mesh
from femsolver.postprocessing.stress_recovery import von_mises

# ---------------------------------------------------------------------------
# Element type → edge index pairs (local node indices within element)
# ---------------------------------------------------------------------------
_EDGES: Dict[str, List[Tuple[int, int]]] = {
    "bar1d":         [(0, 1)],
    "bar2d":         [(0, 1)],
    "bar1d_nl":      [(0, 1)],
    "rod1d_thermal": [(0, 1)],
    "tri3":          [(0, 1), (1, 2), (2, 0)],
    "tri3_thermal":  [(0, 1), (1, 2), (2, 0)],
    "quad4":         [(0, 1), (1, 2), (2, 3), (3, 0)],
    "quad4_axisym":  [(0, 1), (1, 2), (2, 3), (3, 0)],
}
_EDGES_DEFAULT = [(0, 1)]

# Element types rendered as lines (not filled polygons)
_IS_LINE_ELEM = {"bar1d", "bar2d", "bar1d_nl", "rod1d_thermal"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with:  pip install matplotlib"
        )


def _get_cmap(name: str):
    """Return a colormap by name, compatible with both old and new matplotlib."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _node_coords_2d(mesh: Mesh) -> Dict[int, np.ndarray]:
    """Map node_id → 2-D [x, y] array (pad y=0 for 1-D nodes)."""
    coords: Dict[int, np.ndarray] = {}
    for node in mesh.nodes:
        c = node.coords
        if len(c) >= 2:
            coords[node.id] = np.array([c[0], c[1]], dtype=float)
        else:
            coords[node.id] = np.array([c[0], 0.0], dtype=float)
    return coords



def _draw_mesh_edges(
    ax,
    mesh: Mesh,
    coords: Dict[int, np.ndarray],
    **line_kwargs,
) -> None:
    """Draw all element edges onto *ax* using *coords*."""
    for elem in mesh.elements:
        edges = _EDGES.get(elem.type, _EDGES_DEFAULT)
        nids = elem.node_ids
        for i, j in edges:
            if i < len(nids) and j < len(nids):
                p1 = coords[nids[i]]
                p2 = coords[nids[j]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **line_kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_structural(
    mesh: Mesh,
    result_data: dict,
    scale: float = 1.0,
    ax=None,
):
    """Plot undeformed + deformed mesh coloured by von Mises stress.

    Parameters
    ----------
    mesh        : Mesh object from build_model
    result_data : dict parsed from the result YAML
    scale       : displacement magnification factor
    ax          : existing matplotlib Axes or None (a new figure is created)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection, LineCollection
    plt = _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    orig_coords = _node_coords_2d(mesh)

    # --- nodal displacements ---
    disp_section = result_data.get("nodal_displacements", {})
    disp_map: Dict[int, np.ndarray] = {}
    for row in disp_section.get("data", []):
        nid = int(row[0])
        vals = [float(v) for v in row[1:]]
        ux = vals[0] if len(vals) > 0 else 0.0
        uy = vals[1] if len(vals) > 1 else 0.0
        disp_map[nid] = np.array([ux, uy])

    def_coords: Dict[int, np.ndarray] = {
        nid: orig_coords[nid] + scale * disp_map.get(nid, np.zeros(2))
        for nid in orig_coords
    }

    # --- element stresses → von Mises ---
    stress_map: Dict[int, List[float]] = {}
    for row in result_data.get("element_stresses", {}).get("data", []):
        eid = int(row[0])
        stress_map[eid] = [float(v) for v in row[1:]]

    vm_values: Dict[int, float] = {
        elem.id: von_mises(np.array(stress_map.get(elem.id, [0.0]), dtype=float))
        for elem in mesh.elements
    }

    # Shared normalisation across all element types
    all_vm = list(vm_values.values()) or [0.0]
    vm_min, vm_max = min(all_vm), max(all_vm)
    if vm_max == vm_min:
        vm_max = vm_min + 1.0
    cmap = _get_cmap("jet")
    norm = mcolors.Normalize(vmin=vm_min, vmax=vm_max)

    # 1. Undeformed mesh — light grey dashed
    _draw_mesh_edges(ax, mesh, orig_coords,
                     color="lightgrey", linestyle="--", linewidth=0.8, zorder=1)

    # 2. Fill 2-D elements with von Mises colour (deformed positions)
    poly_patches: List[Polygon] = []
    poly_colors: List[float] = []
    for elem in mesh.elements:
        if elem.type in _IS_LINE_ELEM:
            continue
        pts = np.array([def_coords[nid] for nid in elem.node_ids])
        poly_patches.append(Polygon(pts, closed=True))
        poly_colors.append(vm_values[elem.id])

    has_cbar = False
    if poly_patches:
        pc = PatchCollection(poly_patches, cmap=cmap, norm=norm, alpha=0.7, zorder=2)
        pc.set_array(np.array(poly_colors))
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="von Mises stress [Pa]")
        has_cbar = True

    # 3. Draw 1-D elements as coloured lines (deformed positions)
    line_segs: List[List[np.ndarray]] = []
    line_vals: List[float] = []
    for elem in mesh.elements:
        if elem.type not in _IS_LINE_ELEM:
            continue
        nids = elem.node_ids
        line_segs.append([def_coords[nids[0]], def_coords[nids[1]]])
        line_vals.append(vm_values[elem.id])

    if line_segs:
        lc = LineCollection(line_segs, cmap=cmap, norm=norm,
                             linewidths=3.0, alpha=0.9, zorder=3)
        lc.set_array(np.array(line_vals))
        ax.add_collection(lc)
        if not has_cbar:
            fig.colorbar(lc, ax=ax, label="von Mises stress [Pa]")

    # 4. Deformed mesh edges — black solid
    _draw_mesh_edges(ax, mesh, def_coords,
                     color="black", linestyle="-", linewidth=0.8, zorder=4)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Deformed mesh (scale×{scale:.3g}) — von Mises stress")
    plt.tight_layout()
    return fig


def plot_thermal(
    mesh: Mesh,
    result_data: dict,
    ax=None,
):
    """Plot mesh coloured by nodal temperature.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection, LineCollection
    plt = _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    orig_coords = _node_coords_2d(mesh)

    # --- nodal temperatures ---
    temp_map: Dict[int, float] = {}
    for row in result_data.get("nodal_temperatures", {}).get("data", []):
        nid = int(row[0])
        temp_map[nid] = float(row[1])

    # Element-average temperature for fill colour
    elem_temp: Dict[int, float] = {
        elem.id: float(np.mean([temp_map.get(nid, 0.0) for nid in elem.node_ids]))
        for elem in mesh.elements
    }

    all_temps = list(elem_temp.values()) or [0.0]
    t_min, t_max = min(all_temps), max(all_temps)
    if t_max == t_min:
        t_max = t_min + 1.0
    cmap = _get_cmap("hot")
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)

    # Fill 2-D elements
    poly_patches: List[Polygon] = []
    poly_colors: List[float] = []
    for elem in mesh.elements:
        if elem.type in _IS_LINE_ELEM:
            continue
        pts = np.array([orig_coords[nid] for nid in elem.node_ids])
        poly_patches.append(Polygon(pts, closed=True))
        poly_colors.append(elem_temp[elem.id])

    has_cbar = False
    if poly_patches:
        pc = PatchCollection(poly_patches, cmap=cmap, norm=norm, alpha=0.8, zorder=2)
        pc.set_array(np.array(poly_colors))
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="Temperature [K]")
        has_cbar = True

    # Coloured lines for 1-D thermal/bar elements
    line_segs: List[List[np.ndarray]] = []
    line_vals: List[float] = []
    for elem in mesh.elements:
        if elem.type not in _IS_LINE_ELEM:
            continue
        nids = elem.node_ids
        line_segs.append([orig_coords[nids[0]], orig_coords[nids[1]]])
        line_vals.append(elem_temp[elem.id])

    if line_segs:
        lc = LineCollection(line_segs, cmap=cmap, norm=norm,
                             linewidths=4.0, alpha=0.9, zorder=3)
        lc.set_array(np.array(line_vals))
        ax.add_collection(lc)
        if not has_cbar:
            fig.colorbar(lc, ax=ax, label="Temperature [K]")

    # Mesh edges
    _draw_mesh_edges(ax, mesh, orig_coords,
                     color="grey", linestyle="-", linewidth=0.8, zorder=4)

    # Scatter nodes coloured by temperature
    if temp_map:
        ts = [temp_map[nid] for nid in temp_map]
        xs = [orig_coords[nid][0] for nid in temp_map]
        ys = [orig_coords[nid][1] for nid in temp_map]
        t_node_min, t_node_max = min(ts), max(ts)
        if t_node_max == t_node_min:
            t_node_max = t_node_min + 1.0
        node_norm = mcolors.Normalize(vmin=t_node_min, vmax=t_node_max)
        ax.scatter(xs, ys, c=ts, cmap=cmap, norm=node_norm,
                   s=40, zorder=5, edgecolors="k", linewidths=0.5)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Thermal steady-state — temperature field")
    plt.tight_layout()
    return fig

def plot_axisym_3d(
    mesh: Mesh,
    result_data: dict,
    scale: float = 1.0,
    n_theta: int = 60,
    ax=None,
):
    """Revolve axisymmetric (r, z) result into a 3D surface plot.

    Parameters
    ----------
    mesh        : Mesh with nodes in (r, z) coordinates
    result_data : dict parsed from the result YAML (fields: ur, uz)
    scale       : displacement magnification factor
    n_theta     : number of angular divisions for revolution
    ax          : existing 3D Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.colors as mcolors
    from matplotlib.colors import LightSource
    plt = _require_mpl()

    # --- node coordinates (r, z) ---
    node_r: Dict[int, float] = {}
    node_z: Dict[int, float] = {}
    for node in mesh.nodes:
        c = node.coords
        node_r[node.id] = float(c[0])
        node_z[node.id] = float(c[1]) if len(c) > 1 else 0.0

    # --- nodal displacements (ur, uz) ---
    disp_section = result_data.get("nodal_displacements", {})
    disp_ur: Dict[int, float] = {}
    disp_uz: Dict[int, float] = {}
    for row in disp_section.get("data", []):
        nid = int(row[0])
        disp_ur[nid] = float(row[1]) if len(row) > 1 else 0.0
        disp_uz[nid] = float(row[2]) if len(row) > 2 else 0.0

    # --- deformed coordinates ---
    r_def: Dict[int, float] = {}
    z_def: Dict[int, float] = {}
    for nid in node_r:
        r_def[nid] = node_r[nid] + scale * disp_ur.get(nid, 0.0)
        z_def[nid] = node_z[nid] + scale * disp_uz.get(nid, 0.0)

    # --- identify top and bottom surface nodes ---
    # Group nodes by z-coordinate (with tolerance)
    all_z = np.array([node_z[nid] for nid in node_z])
    z_tol = (all_z.max() - all_z.min()) * 1e-6 if all_z.max() != all_z.min() else 1e-12
    z_max_val = all_z.max()
    z_min_val = all_z.min()

    top_nids = sorted(
        [nid for nid in node_z if abs(node_z[nid] - z_max_val) < z_tol],
        key=lambda nid: node_r[nid],
    )
    bot_nids = sorted(
        [nid for nid in node_z if abs(node_z[nid] - z_min_val) < z_tol],
        key=lambda nid: node_r[nid],
    )

    # --- revolve surfaces ---
    theta = np.linspace(0, 2 * np.pi, n_theta)

    def _revolve_surface(nids):
        r_arr = np.array([r_def[nid] for nid in nids])
        z_arr = np.array([z_def[nid] for nid in nids])
        uz_arr = np.array([disp_uz.get(nid, 0.0) for nid in nids])

        # shape: (n_radial, n_theta)
        X = r_arr[:, None] * np.cos(theta[None, :])
        Y = r_arr[:, None] * np.sin(theta[None, :])
        Z = z_arr[:, None] * np.ones_like(theta[None, :])
        C = uz_arr[:, None] * np.ones_like(theta[None, :])
        return X, Y, Z, C

    X_top, Y_top, Z_top, C_top = _revolve_surface(top_nids)
    X_bot, Y_bot, Z_bot, C_bot = _revolve_surface(bot_nids)

    # --- colour normalization across both surfaces ---
    all_c = np.concatenate([C_top.ravel(), C_bot.ravel()])
    c_min, c_max = float(all_c.min()), float(all_c.max())
    if c_max == c_min:
        c_max = c_min + 1.0
    cmap = _get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=c_min, vmax=c_max)

    # --- create 3D figure ---
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ls = LightSource(azdeg=300, altdeg=45)
    # Plot top surface
    face_top = cmap(norm(C_top))
    top_shaded = ls.shade_rgb(face_top, elevation=Z_top, blend_mode='soft', fraction=0.3)
    ax.plot_surface(X_top, Y_top, Z_top, facecolors=top_shaded,
                    alpha=0.95, shade=False, rstride=1, cstride=1)

    # Plot bottom surface
    face_bot = cmap(norm(C_bot))
    bot_shaded = ls.shade_rgb(face_bot, elevation=Z_bot, blend_mode='soft', fraction=0.3)
    ax.plot_surface(X_bot, Y_bot, Z_bot, facecolors=bot_shaded,
                    alpha=0.85, shade=False, rstride=1, cstride=1)

    # Colorbar via ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="uz displacement [m]", shrink=0.6, pad=0.1)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]", labelpad=15)
    ax.tick_params(axis="z", pad=10)
    ax.set_title(f"Axisymmetric 3D revolve (scale×{scale:.3g})")

    # Equal aspect ratio for all axes
    all_X = np.concatenate([X_top.ravel(), X_bot.ravel()])
    all_Y = np.concatenate([Y_top.ravel(), Y_bot.ravel()])
    all_Z = np.concatenate([Z_top.ravel(), Z_bot.ravel()])
    max_range = max(all_X.max() - all_X.min(),
                    all_Y.max() - all_Y.min(),
                    all_Z.max() - all_Z.min()) / 2.0
    mid_x = (all_X.max() + all_X.min()) / 2.0
    mid_y = (all_Y.max() + all_Y.min()) / 2.0
    mid_z = (all_Z.max() + all_Z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig

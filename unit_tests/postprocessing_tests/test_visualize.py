"""Headless integration tests for femsolver/postprocessing/plot_results.py.

Uses matplotlib Agg backend (no display required).
"""
import matplotlib
matplotlib.use("Agg")  # must be set before any pyplot import

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from femsolver.core.mesh import Element, Mesh, Node
from femsolver.postprocessing.plot_results import (
    plot_structural,
    plot_thermal,
)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _bar1d_mesh():
    """3-node, 2-element 1-D bar mesh."""
    nodes = [Node(i + 1, np.array([float(i)])) for i in range(3)]
    elems = [
        Element(1, "bar1d", [1, 2], "m"),
        Element(2, "bar1d", [2, 3], "m"),
    ]
    return Mesh(nodes=nodes, elements=elems)


def _quad4_mesh():
    """4-node, 1-element quad4 mesh (unit square)."""
    nodes = [
        Node(1, np.array([0.0, 0.0])),
        Node(2, np.array([1.0, 0.0])),
        Node(3, np.array([1.0, 1.0])),
        Node(4, np.array([0.0, 1.0])),
    ]
    elems = [Element(1, "quad4", [1, 2, 3, 4], "m")]
    return Mesh(nodes=nodes, elements=elems)


def _thermal_mesh():
    """5-node, 4-element 1-D rod mesh."""
    nodes = [Node(i + 1, np.array([i * 0.25])) for i in range(5)]
    elems = [Element(i + 1, "rod1d_thermal", [i + 1, i + 2], "m") for i in range(4)]
    return Mesh(nodes=nodes, elements=elems)


# ---------------------------------------------------------------------------
# Result-data stubs
# ---------------------------------------------------------------------------

def _structural_bar_result():
    return {
        "metadata": {"problem_type": "structural_static"},
        "nodal_displacements": {
            "units": "m",
            "fields": ["ux"],
            "data": [[1, 0.0], [2, 5e-5], [3, 1e-4]],
        },
        "element_stresses": {
            "units": "Pa",
            "data": [[1, 1e6], [2, 2e6]],
        },
    }


def _structural_quad_result():
    return {
        "metadata": {"problem_type": "structural_static"},
        "nodal_displacements": {
            "units": "m",
            "fields": ["ux", "uy"],
            "data": [
                [1, 0.0,  0.0],
                [2, 1e-5, 0.0],
                [3, 1e-5, 1e-5],
                [4, 0.0,  1e-5],
            ],
        },
        "element_stresses": {
            "units": "Pa",
            "data": [[1, 1e6, 5e5, 2e5]],
        },
    }


def _thermal_result():
    return {
        "metadata": {"problem_type": "thermal_steady"},
        "nodal_temperatures": {
            "units": "K",
            "fields": ["T"],
            "data": [[1, 0.0], [2, 25.0], [3, 50.0], [4, 75.0], [5, 100.0]],
        },
    }


def _modal_result():
    return {
        "metadata": {"problem_type": "modal"},
        "modal_results": {
            "n_modes": 3,
            "modes": [
                {
                    "mode": 1,
                    "frequency_hz": 100.0,
                    "shape": {
                        "fields": ["ux"],
                        "data": [[1, 0.0], [2, 0.71], [3, 1.0]],
                    },
                },
                {
                    "mode": 2,
                    "frequency_hz": 300.0,
                    "shape": {
                        "fields": ["ux"],
                        "data": [[1, 0.0], [2, -1.0], [3, 0.0]],
                    },
                },
                {
                    "mode": 3,
                    "frequency_hz": 500.0,
                    "shape": {
                        "fields": ["ux"],
                        "data": [[1, 0.0], [2, 0.71], [3, -1.0]],
                    },
                },
            ],
        },
    }


def _dynamic_result():
    return {
        "metadata": {"problem_type": "structural_dynamic"},
        "time_history": {
            "units": "m",
            "fields": ["ux", "uy"],
            "n_steps": 3,
            "steps": [
                {"time": 0.0,   "data": [[1, 0.0,  0.0], [2, 0.0,  0.0]]},
                {"time": 0.001, "data": [[1, 1e-5, 0.0], [2, 2e-5, 0.0]]},
                {"time": 0.002, "data": [[1, 2e-5, 0.0], [2, 4e-5, 0.0]]},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Tests — plot_structural
# ---------------------------------------------------------------------------

def test_structural_static_plot():
    """plot_structural returns a Figure for a 1-D bar mesh."""
    fig = plot_structural(_bar1d_mesh(), _structural_bar_result(), scale=1.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_structural_static_plot_2d():
    """plot_structural returns a Figure for a 2-D quad4 mesh."""
    fig = plot_structural(_quad4_mesh(), _structural_quad_result(), scale=100.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_structural_plot_with_ax():
    """plot_structural accepts a pre-existing Axes object."""
    fig_in, ax = plt.subplots()
    fig_out = plot_structural(_bar1d_mesh(), _structural_bar_result(), ax=ax)
    assert fig_out is fig_in
    plt.close(fig_in)


def test_structural_plot_no_stress_data():
    """plot_structural handles empty element_stresses gracefully."""
    result = {
        "metadata": {"problem_type": "structural_static"},
        "nodal_displacements": {
            "units": "m",
            "fields": ["ux"],
            "data": [[1, 0.0], [2, 1e-4]],
        },
        "element_stresses": {"units": "Pa", "data": []},
    }
    mesh = Mesh(
        nodes=[Node(1, np.array([0.0])), Node(2, np.array([1.0]))],
        elements=[Element(1, "bar1d", [1, 2], "m")],
    )
    fig = plot_structural(mesh, result)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — plot_thermal
# ---------------------------------------------------------------------------

def test_thermal_plot():
    """plot_thermal returns a Figure for a 1-D rod mesh."""
    fig = plot_thermal(_thermal_mesh(), _thermal_result())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_thermal_plot_with_ax():
    """plot_thermal accepts a pre-existing Axes object."""
    fig_in, ax = plt.subplots()
    fig_out = plot_thermal(_thermal_mesh(), _thermal_result(), ax=ax)
    assert fig_out is fig_in
    plt.close(fig_in)
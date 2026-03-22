"""Unit tests for the Tri3 (CST) element."""
import numpy as np
import pytest

from femsolver.elements.tri3 import Tri3Element
from femsolver.materials.linear_elastic import LinearElasticMaterial


def _mat(E=1.0, nu=0.0, rho=1.0, thickness=1.0):
    return LinearElasticMaterial("m", E=E, nu=nu, rho=rho, thickness=thickness)


def _unit_tri(plane="stress"):
    """Right triangle with nodes (0,0),(1,0),(0,1)."""
    mat = _mat()
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return Tri3Element(1, [1, 2, 3], coords, mat, plane=plane)


def _rect_tri(plane="stress"):
    """Right triangle from (0,0),(2,0),(0,1), area=1."""
    mat = _mat()
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    return Tri3Element(1, [1, 2, 3], coords, mat, plane=plane)


# --- Geometry checks ---

def test_unit_tri_area():
    tri = _unit_tri()
    assert tri.area == pytest.approx(0.5, rel=1e-12)


def test_zero_area_raises():
    mat = _mat()
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # collinear
    with pytest.raises(ValueError, match="zero"):
        Tri3Element(1, [1, 2, 3], coords, mat)


# --- B matrix ---

def test_B_matrix_shape():
    tri = _unit_tri()
    B = tri.compute_B_matrix()
    assert B.shape == (3, 6)


def test_B_matrix_pure_axial_strain():
    """For u_x = \epsilon*x, u_y = 0 on unit tri: \epsilon_xx=\epsilon, \epsilon_yy=0, \gamma_xy=0."""
    tri = _unit_tri()
    eps = 0.01
    # Nodes (0,0),(1,0),(0,1) → ux = [0, eps, 0], uy = [0, 0, 0]
    u_e = np.array([0.0, 0.0, eps, 0.0, 0.0, 0.0])
    B = tri.compute_B_matrix()
    strain = B @ u_e
    np.testing.assert_allclose(strain[0], eps, rtol=1e-12)    # \epsilon_xx
    np.testing.assert_allclose(strain[1], 0.0, atol=1e-14)    # \epsilon_yy
    np.testing.assert_allclose(strain[2], 0.0, atol=1e-14)    # \gamma_xy


def test_B_matrix_pure_shear_strain():
    """For u_x = 0, u_y = \gamma * x: \epsilon_xx=0, \epsilon_yy=0, \gamma_xy=\gamma."""
    tri = _unit_tri()
    gamma = 0.005
    # u_y = \gamma * x → at (0,0): uy=0; at (1,0): uy=\gamma; at (0,1): uy=0
    u_e = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
    B = tri.compute_B_matrix()
    strain = B @ u_e
    np.testing.assert_allclose(strain[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(strain[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(strain[2], gamma, rtol=1e-12)


# --- Patch test: rigid body motions produce zero strain ---

def test_rigid_translation_x_zero_strain():
    tri = _unit_tri()
    u_e = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # ux = const
    B = tri.compute_B_matrix()
    np.testing.assert_allclose(B @ u_e, np.zeros(3), atol=1e-14)


def test_rigid_translation_y_zero_strain():
    tri = _unit_tri()
    u_e = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])  # uy = const
    B = tri.compute_B_matrix()
    np.testing.assert_allclose(B @ u_e, np.zeros(3), atol=1e-14)


# --- Stiffness matrix ---

def test_stiffness_matrix_shape():
    tri = _unit_tri()
    k = tri.compute_k_e()
    assert k.shape == (6, 6)


def test_stiffness_matrix_symmetric():
    tri = _unit_tri()
    k = tri.compute_k_e()
    np.testing.assert_allclose(k, k.T, atol=1e-14)


def test_stiffness_matrix_positive_semidefinite():
    """Has 3 zero eigenvalues (rigid body modes) and all others positive."""
    tri = _unit_tri()
    k = tri.compute_k_e()
    eigvals = np.linalg.eigvalsh(k)
    # 3 rigid body modes → near-zero eigenvalues
    assert np.sum(eigvals < -1e-10) == 0, f"Negative eigenvalue found: {eigvals}"
    assert np.sum(np.abs(eigvals) < 1e-10) == 3, \
        f"Expected 3 zero eigenvalues, got: {eigvals}"


# --- Stress recovery ---

def test_stress_uniaxial_plane_stress():
    """\sigma_xx = E·\epsilon, \sigma_yy = 0, \tau_xy = 0 for pure axial strain (ν=0)."""
    E, eps = 200.0, 0.001
    mat = LinearElasticMaterial("m", E=E, nu=0.0, rho=1.0, thickness=1.0)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = Tri3Element(1, [1, 2, 3], coords, mat, plane="stress")
    u_e = np.array([0.0, 0.0, eps, 0.0, 0.0, 0.0])
    sigma = tri.compute_stress(u_e)
    np.testing.assert_allclose(sigma[0], E * eps, rtol=1e-12)
    np.testing.assert_allclose(sigma[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(sigma[2], 0.0, atol=1e-10)

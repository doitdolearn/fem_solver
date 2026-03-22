"""Unit tests for the Quad4 bilinear isoparametric element."""
import numpy as np
import pytest

from femsolver.elements.quad4 import Quad4Element
from femsolver.materials.linear_elastic import LinearElasticMaterial


def _mat(E=1.0, nu=0.0, rho=1.0, thickness=1.0):
    return LinearElasticMaterial("m", E=E, nu=nu, rho=rho, thickness=thickness)


def _unit_quad(plane="stress"):
    """Unit square with nodes at (0,0),(1,0),(1,1),(0,1) — CCW."""
    mat = _mat()
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Quad4Element(1, [1, 2, 3, 4], coords, mat, plane=plane)


def _rect_quad(lx=2.0, ly=3.0, plane="stress"):
    """Rectangle (0,0)→(lx,0)→(lx,ly)→(0,ly)."""
    mat = _mat()
    coords = np.array([[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]])
    return Quad4Element(1, [1, 2, 3, 4], coords, mat, plane=plane)


# --- Shape functions ---

def test_shape_functions_sum_to_one():
    q = _unit_quad()
    for r in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for s in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            N = q.shape_functions(r, s)
            assert abs(N.sum() - 1.0) < 1e-14, f"Sum(N) ≠ 1 at (r={r},s={s})"


def test_shape_functions_at_corner_nodes():
    q = _unit_quad()
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for k, (r, s) in enumerate(corners):
        N = q.shape_functions(r, s)
        expected = np.zeros(4)
        expected[k] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-14)


# --- Jacobian ---

def test_jacobian_positive_unit_square():
    q = _unit_quad()
    for r in [-0.5, 0.0, 0.5]:
        for s in [-0.5, 0.0, 0.5]:
            J = q._jacobian(r, s)
            assert np.linalg.det(J) > 0


def test_jacobian_determinant_unit_square():
    """For unit square, det(J) = 1/4 everywhere."""
    q = _unit_quad()
    for r in [-0.5, 0.0, 0.5]:
        for s in [-0.5, 0.0, 0.5]:
            J = q._jacobian(r, s)
            np.testing.assert_allclose(np.linalg.det(J), 0.25, rtol=1e-12)


def test_jacobian_determinant_rectangle():
    """For lx×ly rectangle, det(J) = lx*ly/4."""
    lx, ly = 3.0, 5.0
    q = _rect_quad(lx=lx, ly=ly)
    J = q._jacobian(0.0, 0.0)
    np.testing.assert_allclose(np.linalg.det(J), lx * ly / 4.0, rtol=1e-12)


# --- B matrix ---

def test_B_matrix_shape():
    q = _unit_quad()
    B = q.compute_B_matrix()
    assert B.shape == (3, 8)


def test_B_matrix_patch_test_axial():
    """u_x = ε·x, u_y = 0 on unit square → εxx=ε, εyy=0, γxy=0."""
    q = _unit_quad()
    eps = 0.002
    # Nodes (0,0),(1,0),(1,1),(0,1): ux = [0, eps, eps, 0], uy = [0,0,0,0]
    u_e = np.array([0.0, 0.0, eps, 0.0, eps, 0.0, 0.0, 0.0])
    B = q.compute_B_matrix()    # centroid
    strain = B @ u_e
    np.testing.assert_allclose(strain[0], eps, rtol=1e-12)
    np.testing.assert_allclose(strain[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(strain[2], 0.0, atol=1e-14)


def test_B_matrix_patch_test_transverse():
    """u_x = 0, u_y = ε·y on unit square → εxx=0, εyy=ε, γxy=0."""
    q = _unit_quad()
    eps = 0.003
    # Nodes: uy = [0, 0, eps, eps]
    u_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, eps, 0.0, eps])
    B = q.compute_B_matrix()
    strain = B @ u_e
    np.testing.assert_allclose(strain[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(strain[1], eps, rtol=1e-12)
    np.testing.assert_allclose(strain[2], 0.0, atol=1e-14)


def test_rigid_translation_zero_strain():
    q = _unit_quad()
    for delta in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        u_e = np.tile(delta, 4)
        B = q.compute_B_matrix()
        np.testing.assert_allclose(B @ u_e, np.zeros(3), atol=1e-14,
                                    err_msg=f"Translation {delta} should give zero strain")


# --- Stiffness matrix ---

def test_stiffness_matrix_shape():
    q = _unit_quad()
    k = q.compute_k_e()
    assert k.shape == (8, 8)


def test_stiffness_matrix_symmetric():
    q = _unit_quad()
    k = q.compute_k_e()
    np.testing.assert_allclose(k, k.T, atol=1e-13)


def test_stiffness_matrix_positive_semidefinite():
    """3 rigid body modes → 3 near-zero eigenvalues; rest positive."""
    q = _unit_quad()
    k = q.compute_k_e()
    eigvals = np.linalg.eigvalsh(k)
    assert np.sum(eigvals < -1e-10) == 0, f"Negative eigenvalue: {min(eigvals):.3e}"
    assert np.sum(np.abs(eigvals) < 1e-10) == 3


def test_stiffness_scales_with_thickness():
    E, t1, t2 = 1.0, 1.0, 3.0
    mat1 = LinearElasticMaterial("m1", E=E, nu=0.0, rho=1.0, thickness=t1)
    mat2 = LinearElasticMaterial("m2", E=E, nu=0.0, rho=1.0, thickness=t2)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    q1 = Quad4Element(1, [1, 2, 3, 4], coords, mat1)
    q2 = Quad4Element(1, [1, 2, 3, 4], coords, mat2)
    np.testing.assert_allclose(q2.compute_k_e(), (t2 / t1) * q1.compute_k_e(),
                               rtol=1e-10, atol=1e-14)


# --- Stress recovery ---

def test_stress_uniaxial_plane_stress():
    """σxx = E·ε, σyy = 0, τxy = 0 (ν=0, pure axial strain)."""
    E, eps = 200.0, 0.001
    mat = LinearElasticMaterial("m", E=E, nu=0.0, rho=1.0, thickness=1.0)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    q = Quad4Element(1, [1, 2, 3, 4], coords, mat, plane="stress")
    u_e = np.array([0.0, 0.0, eps, 0.0, eps, 0.0, 0.0, 0.0])
    sigma = q.compute_stress(u_e)
    np.testing.assert_allclose(sigma[0], E * eps, rtol=1e-12)
    np.testing.assert_allclose(sigma[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(sigma[2], 0.0, atol=1e-10)


# --- Non-positive Jacobian ---

def test_non_positive_jacobian_raises():
    """CW node ordering gives negative Jacobian → should raise."""
    mat = _mat()
    # Reversed (CW) ordering: (0,0),(0,1),(1,1),(1,0)
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    q = Quad4Element(1, [1, 2, 3, 4], coords, mat)
    with pytest.raises(ValueError, match="[Jj]acobian"):
        q.compute_k_e()

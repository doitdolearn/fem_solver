import numpy as np
import pytest

from femsolver.elements.bar1d import Bar1DElement
from femsolver.materials.linear_elastic import LinearElasticMaterial


def _mat(E=200e9, A=0.01, rho=7850.0):
    return LinearElasticMaterial("steel", E=E, nu=0.3, rho=rho, A=A)


def _bar(L=1.0, E=200e9, A=0.01, rho=7850.0):
    mat = _mat(E=E, A=A, rho=rho)
    coords = np.array([[0.0], [L]])
    return Bar1DElement(1, [1, 2], coords, mat)


def test_stiffness_matrix():
    E, A, L = 200e9, 0.01, 1.0
    bar = _bar(L=L, E=E, A=A)
    k = bar.compute_k_e()
    expected = (E * A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(k, expected, rtol=1e-12)


def test_stiffness_scales_with_EA_over_L():
    bar = _bar(L=2.0, E=100.0, A=5.0)
    k = bar.compute_k_e()
    c = 100.0 * 5.0 / 2.0
    expected = c * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(k, expected, rtol=1e-12)


def test_body_force_vector_is_zero():
    bar = _bar()
    f = bar.compute_f_e()
    np.testing.assert_array_equal(f, np.zeros(2))


def test_zero_length_raises():
    mat = _mat()
    coords = np.array([[0.0], [0.0]])
    with pytest.raises(ValueError, match="zero"):
        Bar1DElement(1, [1, 2], coords, mat)


def test_stress_recovery_tensile():
    E, L = 200e9, 1.0
    bar = _bar(L=L, E=E)
    u_e = np.array([0.0, 1e-3])
    stress = bar.compute_stress(u_e)
    expected = E * 1e-3 / L
    np.testing.assert_allclose(stress[0], expected, rtol=1e-12)


def test_stress_recovery_compressive():
    E, L = 200e9, 1.0
    bar = _bar(L=L, E=E)
    u_e = np.array([1e-3, 0.0])
    stress = bar.compute_stress(u_e)
    expected = -E * 1e-3 / L
    np.testing.assert_allclose(stress[0], expected, rtol=1e-12)


def test_B_matrix_shape_and_values():
    L = 2.0
    bar = _bar(L=L)
    B = bar.compute_B_matrix()
    assert B.shape == (1, 2)
    np.testing.assert_allclose(B[0, 0], -1.0 / L, rtol=1e-12)
    np.testing.assert_allclose(B[0, 1],  1.0 / L, rtol=1e-12)


def test_shape_functions_sum_to_one():
    bar = _bar()
    for r in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        N = bar.shape_functions(r)
        assert abs(N.sum() - 1.0) < 1e-14

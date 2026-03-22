from __future__ import annotations

from typing import Tuple

import numpy as np

def gauss_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Return (points, weights) for n point Gaussian integration points on [-1, 1]
    Returns:
        points: size n array
        weights: size n array
    """

    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    elif n == 3:
        a = np.sqrt(3.0 / 5.0)
        return np.array([-a, 0, a]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f"Unsupported n={n}. Use 1, 2, or 3.")

def gauss_2d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Return (points, weights) for n point Gaussian integration points on [-1, 1]^2

    Returns:
        points: size [n^2, 2] array of (r,s) coordinates
        weights: size [n^2, ] array
    """
    points_1d, weights_1d = gauss_1d(n)
    points_2d = []
    weights_2d = []
    for ri, wi in zip(points_1d, weights_1d):
        for sj, wj in zip(points_1d, weights_1d):
            points_2d.append([ri, sj])
            weights_2d.append(wi * wj)
    return np.array(points_2d), np.array(weights_2d)

def gauss_3d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Return (points, weights) for n point Gaussian integration points on [-1, 1]^3

    Returns:
        points: size [n^3, 2] array of (r,s,t) coordinates
        weights: size [n^3, ] array
    """
    points_1d, weights_1d = gauss_1d(n)
    points_3d = []
    weights_3d = []
    for ri, wi in zip(points_1d, weights_1d):
        for sj, wj in zip(points_1d, weights_1d):
            for tk, wk in zip(points_1d, weights_1d):
                points_3d.append([ri, sj, tk])
                weights_3d.append(wi * wj * wk)
    return np.array(points_3d), np.array(weights_3d)

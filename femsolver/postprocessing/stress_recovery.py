from __future__ import annotations

import numpy as np

def von_mises(sigma: np.ndarray) -> float:
    """Scalar stress measure from a Voigt stress vector."""
    if len(sigma) == 1:
        return abs(float(sigma[0]))
    if len(sigma) == 4:
        # Axisymmetric: [s_rr, s_tt, s_zz, tau_rz]
        s1, s2, s3 = float(sigma[0]), float(sigma[1]), float(sigma[2])
        t31 = float(sigma[3])
        val = 0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2) + \
              3.0 * t31**2
        return float(np.sqrt(max(val, 0.0)))
    if len(sigma) == 6:
        # 3D: [s_rr, s_tt, s_zz, tau_rt, tau_tz, tau_zr]
        s1, s2, s3 = float(sigma[0]), float(sigma[1]), float(sigma[2])
        t12, t23, t31 = float(sigma[3]), float(sigma[4]), float(sigma[5])
        val = 0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2) + \
              3.0 * (t12**2 + t23**2 + t31**2)
        return float(np.sqrt(max(val, 0.0)))
    sxx, syy, sxy = float(sigma[0]), float(sigma[1]), float(sigma[2])
    return float(np.sqrt(max(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0)))
# sim/metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def angle_rmse_deg(y_true, y_pred):
    """
    Matches predictions to truths via Hungarian assignment on absolute angle error.
    Returns RMSE in degrees and per-pair errors (deg).
    """
    t = np.sort(np.asarray(y_true, dtype=float))
    p = np.sort(np.asarray(y_pred, dtype=float))
    if t.size == 0 or p.size == 0:
        return np.nan, np.array([])

    # Cost matrix (abs error)
    T, P = t.size, p.size
    C = np.zeros((T, P), dtype=float)
    for i in range(T):
        for j in range(P):
            C[i, j] = np.abs(t[i] - p[j])

    r, c = linear_sum_assignment(C)
    errs = C[r, c]
    rmse = np.sqrt(np.mean(errs**2))
    return rmse, errs

def resolved_indicator(y_true, y_pred, threshold_deg=1.0):
    """
    Returns 1 if:
      - number of predicted peaks >= number of true sources, and
      - all matched absolute errors <= threshold_deg
    else 0.
    """
    t = np.sort(np.asarray(y_true, dtype=float))
    p = np.sort(np.asarray(y_pred, dtype=float))
    if p.size < t.size or t.size == 0:
        return 0
    # Match
    rmse, errs = angle_rmse_deg(t, p[:t.size])
    return int(np.all(errs <= threshold_deg))

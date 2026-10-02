"""Exposure metrics on a netting set.

Definitions follow the Basel counterparty credit risk framework:

    EE(t)   = E[ max(V(t), 0) ]                       expected exposure
    EPE     = time-average of EE over the profile      expected positive exposure
    PFE(t)  = q-th percentile of max(V(t), 0)          potential future exposure
    EEPE    = time-average of the running maximum of EE, capped at 1 year
"""

import numpy as np


def gross_exposure(trade_mtms):
    """Sum of per-trade positive exposures. No netting benefit."""
    return np.sum([np.maximum(V, 0.0) for V in trade_mtms], axis=0)


def netted_mtm(trade_mtms):
    """Net MTM across the netting set (can be negative)."""
    return np.sum(trade_mtms, axis=0)


def netted_exposure(trade_mtms):
    return np.maximum(netted_mtm(trade_mtms), 0.0)


def expected_exposure(exposure):
    return exposure.mean(axis=0)


def pfe(exposure, quantile=0.975):
    return np.quantile(exposure, quantile, axis=0)


def epe(exposure, grid):
    ee = expected_exposure(exposure)
    grid = np.asarray(grid, dtype=float)
    return float(np.trapezoid(ee, grid) / (grid[-1] - grid[0]))


def eepe(exposure, grid, horizon=1.0):
    """Effective EPE: average of the running max of EE over the first year."""
    grid = np.asarray(grid, dtype=float)
    ee = expected_exposure(exposure)
    eee = np.maximum.accumulate(ee)
    mask = grid <= horizon + 1e-9
    return float(np.trapezoid(eee[mask], grid[mask]) / (grid[mask][-1] - grid[0]))


def summary(exposure, grid, quantile=0.975):
    ee = expected_exposure(exposure)
    p = pfe(exposure, quantile)
    return {
        "EPE": epe(exposure, grid),
        "EEPE": eepe(exposure, grid),
        "peak_EE": float(ee.max()),
        "peak_EE_time": float(grid[int(ee.argmax())]),
        f"peak_PFE_{int(quantile * 1000) / 10:g}pc": float(p.max()),
        "peak_PFE_time": float(grid[int(p.argmax())]),
    }

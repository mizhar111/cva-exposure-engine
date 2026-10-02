"""Hazard rate bootstrap from a CDS term structure, and CVA.

The hazard rate is piecewise constant between quoted CDS tenors. For each
tenor in turn we solve for the hazard rate that sets the CDS par value to zero:

    premium leg = s * sum_i  dt_i * D(t_i) * Q(t_i)
    protection  = (1 - R) * sum_i D(t_i) * [ Q(t_{i-1}) - Q(t_i) ]

Unilateral CVA on the netting set is then

    CVA = (1 - R) * sum_i D(t_i) * EE(t_i) * [ Q(t_{i-1}) - Q(t_i) ]

Wrong-way risk is introduced by tilting the hazard rate with the simulated
short rate. For a payer swap, exposure rises when rates rise, so a positive
beta means the counterparty is more likely to default exactly when we are most
exposed to it.
"""

import numpy as np
from scipy.optimize import brentq


class CreditCurve:
    """Piecewise-constant hazard rate curve bootstrapped from CDS spreads."""

    def __init__(self, tenors, spreads, recovery=0.40, discount=None, freq=0.25):
        self.tenors = np.asarray(tenors, dtype=float)
        self.spreads = np.asarray(spreads, dtype=float)
        self.recovery = recovery
        self.freq = freq
        self.discount = discount if discount is not None else (lambda t: np.exp(-0.042 * t))
        self.hazards = self._bootstrap()

    # ---- survival ---------------------------------------------------------
    def survival(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.ones_like(t)
        for i, ti in enumerate(t):
            integral, prev = 0.0, 0.0
            for k, T_k in enumerate(self.tenors):
                if ti <= T_k:
                    integral += self.hazards[k] * (ti - prev)
                    break
                integral += self.hazards[k] * (T_k - prev)
                prev = T_k
            else:
                integral += self.hazards[-1] * (ti - prev)
            out[i] = np.exp(-integral)
        return out if out.size > 1 else float(out[0])

    # ---- bootstrap --------------------------------------------------------
    def _bootstrap(self):
        self.hazards = np.zeros(len(self.tenors))
        for k, (T_k, s_k) in enumerate(zip(self.tenors, self.spreads)):
            def objective(h, k=k, T_k=T_k, s_k=s_k):
                trial = self.hazards.copy()
                trial[k] = h
                saved, self.hazards = self.hazards, trial
                pv = self._cds_pv(T_k, s_k)
                self.hazards = saved
                return pv

            self.hazards[k] = brentq(objective, 1e-8, 3.0, xtol=1e-12)
        return self.hazards

    def _cds_pv(self, maturity, spread):
        grid = np.arange(self.freq, maturity + 1e-9, self.freq)
        Q = np.asarray(self.survival(grid), dtype=float).ravel()
        Q_prev = np.concatenate([[1.0], Q[:-1]])
        D = self.discount(grid)
        premium = spread * np.sum(self.freq * D * Q)
        protection = (1 - self.recovery) * np.sum(D * (Q_prev - Q))
        return premium - protection


def cva(ee, grid, curve, discount):
    """Unilateral CVA from a discounted expected exposure profile."""
    grid = np.asarray(grid, dtype=float)
    Q = np.asarray(curve.survival(grid), dtype=float).ravel()
    dQ = np.concatenate([[1.0 - Q[0]], Q[:-1] - Q[1:]])
    D = discount(grid)
    ee_mid = np.concatenate([[ee[0]], 0.5 * (ee[1:] + ee[:-1])])
    return float((1 - curve.recovery) * np.sum(D * ee_mid * dQ))


def cva_wrong_way(exposure, grid, r_paths, curve, discount, beta):
    """Path-wise CVA with a hazard rate tilted by the simulated short rate.

        lambda_i(t) = lambda(t) * exp( beta * (r_i(t) - r0) ) / E[ exp(...) ]

    The normalisation keeps the marginal default probability equal to the
    calibrated curve, so any change in CVA is attributable to the correlation
    between exposure and default, not to a shifted default probability.
    """
    grid = np.asarray(grid, dtype=float)
    r0 = r_paths[:, 0].mean()
    tilt = np.exp(beta * (r_paths - r0))
    tilt /= tilt.mean(axis=0, keepdims=True)

    Q_base = np.asarray(curve.survival(grid), dtype=float).ravel()
    lam = np.zeros_like(Q_base)
    lam[1:] = -np.log(Q_base[1:] / Q_base[:-1]) / np.diff(grid)

    dQ_path = np.zeros_like(exposure)
    surv = np.ones(exposure.shape[0])
    for i in range(1, len(grid)):
        dt = grid[i] - grid[i - 1]
        p_def = surv * (1.0 - np.exp(-lam[i] * tilt[:, i] * dt))
        dQ_path[:, i] = p_def
        surv = surv - p_def

    D = discount(grid)
    return float((1 - curve.recovery) * np.mean(np.sum(exposure * dQ_path * D, axis=1)))

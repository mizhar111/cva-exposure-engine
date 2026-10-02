"""Vanilla interest rate swap: schedule construction and path-wise MTM.

Valuation convention (payer swap, pay fixed / receive float), at time t with
the last floating fixing L_k set at reset date T_k and paid at T_{k+1}:

    PV_float(t) = N * [ tau * L_k * P(t, T_{k+1}) + P(t, T_{k+1}) - P(t, T_n) ]
    PV_fixed(t) = N * K * sum_{T_j > t} tau_j * P(t, T_j)
    V(t)        = PV_float(t) - PV_fixed(t)

Carrying the last fixing explicitly is what lets the exposure be computed on a
weekly grid rather than only on reset dates, which in turn is what makes a
10-business-day margin period of risk meaningful in the collateral model.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InterestRateSwap:
    notional: float
    maturity: float
    fixed_rate: float = None          # set to par at t=0 if left as None
    pay_fixed: bool = True            # True = payer swap
    float_freq: float = 0.25          # quarterly floating leg
    fixed_freq: float = 0.5           # semi-annual fixed leg
    label: str = "swap"

    float_dates: np.ndarray = field(init=False)
    fixed_dates: np.ndarray = field(init=False)

    def __post_init__(self):
        n_f = int(round(self.maturity / self.float_freq))
        n_x = int(round(self.maturity / self.fixed_freq))
        self.float_dates = np.round(np.arange(0, n_f + 1) * self.float_freq, 10)
        self.fixed_dates = np.round(np.arange(1, n_x + 1) * self.fixed_freq, 10)

    # ---- par rate ---------------------------------------------------------
    def par_rate(self, model):
        annuity = self.fixed_freq * np.sum(model.P_market(self.fixed_dates))
        return float((1.0 - model.P_market(self.maturity)) / annuity)

    def set_par(self, model):
        self.fixed_rate = self.par_rate(model)
        return self.fixed_rate

    # ---- path-wise valuation ---------------------------------------------
    def mtm(self, model, grid, r):
        """Mark-to-market on every (path, grid point). Shape (n_paths, n_steps)."""
        grid = np.asarray(grid, dtype=float)
        n_paths, n_steps = r.shape
        V = np.zeros((n_paths, n_steps))

        # Floating fixings observed at each reset date, per path.
        fixings = {}
        for T_k in self.float_dates[:-1]:
            idx = int(np.argmin(np.abs(grid - T_k)))
            T_next = T_k + self.float_freq
            P_k = model.bond_price(grid[idx], T_next, r[:, idx])
            fixings[float(T_k)] = (1.0 / P_k - 1.0) / self.float_freq

        for i, t in enumerate(grid):
            if t >= self.maturity - 1e-9:
                V[:, i] = 0.0
                continue

            # Locate the current floating accrual period.
            k = int(np.searchsorted(self.float_dates, t + 1e-9) - 1)
            k = max(k, 0)
            T_k = float(self.float_dates[k])
            T_next = T_k + self.float_freq
            L_k = fixings[T_k]

            P_next = model.bond_price(t, T_next, r[:, i])
            P_end = model.bond_price(t, self.maturity, r[:, i])
            pv_float = self.float_freq * L_k * P_next + P_next - P_end

            remaining = self.fixed_dates[self.fixed_dates > t + 1e-9]
            pv_fixed = np.zeros(n_paths)
            for T_j in remaining:
                pv_fixed += self.fixed_freq * model.bond_price(t, T_j, r[:, i])
            pv_fixed *= self.fixed_rate

            v = self.notional * (pv_float - pv_fixed)
            V[:, i] = v if self.pay_fixed else -v

        return V

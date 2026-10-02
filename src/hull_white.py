"""One-factor Hull-White short rate model.

    dr(t) = (theta(t) - a * r(t)) dt + sigma dW(t)

Calibrated to a flat initial zero curve, which keeps the theta(t) term
analytic and lets us focus on the exposure mechanics rather than curve
stripping. Bond prices use the standard affine form

    P(t,T) = A(t,T) * exp(-B(t,T) * r(t))

with B and A as in Brigo & Mercurio, "Interest Rate Models", section 3.3.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HullWhite:
    a: float = 0.05          # mean reversion speed
    sigma: float = 0.0075    # short rate volatility (absolute, 75bp)
    r0: float = 0.042        # initial short rate / flat curve level

    # ---- initial (market) curve -------------------------------------------
    def P_market(self, T):
        """Initial discount factor P^M(0,T) implied by a flat curve."""
        return np.exp(-self.r0 * np.asarray(T, dtype=float))

    def f_market(self, t):
        """Initial instantaneous forward rate f^M(0,t). Flat curve -> r0."""
        return np.full_like(np.asarray(t, dtype=float), self.r0)

    # ---- affine bond price ------------------------------------------------
    def B(self, t, T):
        tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def A(self, t, T):
        t = np.asarray(t, dtype=float)
        T = np.asarray(T, dtype=float)
        B = self.B(t, T)
        term = (
            B * self.f_market(t)
            - (self.sigma ** 2) / (4.0 * self.a) * (1.0 - np.exp(-2.0 * self.a * t)) * B ** 2
        )
        return (self.P_market(T) / self.P_market(t)) * np.exp(term)

    def bond_price(self, t, T, r):
        """P(t,T) given the short rate r at time t. r may be a path vector."""
        return self.A(t, T) * np.exp(-self.B(t, T) * np.asarray(r))

    # ---- simulation -------------------------------------------------------
    def simulate(self, grid, n_paths, seed=42):
        """Exact-moment Euler scheme for r on the supplied time grid.

        Returns an (n_paths, n_steps) array of short rates, r[:, 0] = r0.
        """
        rng = np.random.default_rng(seed)
        grid = np.asarray(grid, dtype=float)
        n_steps = len(grid)
        r = np.empty((n_paths, n_steps))
        r[:, 0] = self.r0

        for i in range(1, n_steps):
            t0, t1 = grid[i - 1], grid[i]
            dt = t1 - t0
            decay = np.exp(-self.a * dt)

            # E[r(t1) | r(t0)] under the risk-neutral measure with a flat curve.
            alpha0 = self.r0 + (self.sigma ** 2) / (2 * self.a ** 2) * (1 - np.exp(-self.a * t0)) ** 2
            alpha1 = self.r0 + (self.sigma ** 2) / (2 * self.a ** 2) * (1 - np.exp(-self.a * t1)) ** 2
            mean = r[:, i - 1] * decay + alpha1 - alpha0 * decay

            var = (self.sigma ** 2) / (2 * self.a) * (1 - decay ** 2)
            r[:, i] = mean + np.sqrt(var) * rng.standard_normal(n_paths)

        return r

    def numeraire(self, grid, r):
        """Path-wise money market account B(t) via trapezoidal integration of r."""
        grid = np.asarray(grid, dtype=float)
        dt = np.diff(grid)
        integrand = 0.5 * (r[:, 1:] + r[:, :-1]) * dt
        cum = np.cumsum(integrand, axis=1)
        return np.hstack([np.ones((r.shape[0], 1)), np.exp(cum)])

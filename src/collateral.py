"""Variation margin under a two-way CSA.

Collateral held against the counterparty at time t is based on the netting set
value observed one margin period of risk (MPoR) earlier, because that is the
last valuation we could have called and received margin against before a
default is closed out:

    required(t) = max( V(t - MPoR) - threshold, 0 )

The minimum transfer amount makes the balance sticky: a call is only made when
the required balance moves by at least the MTA. Collateralised exposure is then

    E_coll(t) = max( V(t) - C(t), 0 )

so the residual exposure is the gap-risk over the margin period plus the
uncollateralised threshold.
"""

import numpy as np


def collateral_balance(V_net, grid, threshold, mta, mpor, two_way=True):
    """Path-wise collateral held. V_net shape (n_paths, n_steps)."""
    grid = np.asarray(grid, dtype=float)
    dt = grid[1] - grid[0]
    lag = max(int(round(mpor / dt)), 1)

    n_paths, n_steps = V_net.shape
    C = np.zeros((n_paths, n_steps))
    balance = np.zeros(n_paths)

    for i in range(n_steps):
        j = i - lag
        if j < 0:
            C[:, i] = 0.0
            continue

        v_lag = V_net[:, j]
        if two_way:
            required = np.where(
                v_lag > threshold, v_lag - threshold,
                np.where(v_lag < -threshold, v_lag + threshold, 0.0),
            )
        else:
            required = np.maximum(v_lag - threshold, 0.0)

        move = np.abs(required - balance)
        balance = np.where(move >= mta, required, balance)
        C[:, i] = balance

    return C


def collateralised_exposure(V_net, C):
    return np.maximum(V_net - C, 0.0)

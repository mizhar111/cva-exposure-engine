"""Counterparty credit exposure and CVA on a two-swap netting set.

Run:  python run.py [--paths 10000] [--seed 42]

Produces the exposure profiles, the collateral comparison and the CVA numbers
reported in the README, plus the charts in outputs/.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from collateral import collateral_balance, collateralised_exposure  # noqa: E402
from credit import CreditCurve, cva, cva_wrong_way  # noqa: E402
from exposure import (  # noqa: E402
    expected_exposure, gross_exposure, netted_exposure, netted_mtm, pfe, summary,
)
from hull_white import HullWhite  # noqa: E402
from swap import InterestRateSwap  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "outputs")
PLOT_KW = dict(dpi=140, bbox_inches="tight")

# CSA terms
THRESHOLD = 1_000_000.0
MTA = 250_000.0
MPOR = 10 / 252  # 10 business days

# Counterparty CDS quotes (BBB industrial, bp)
CDS_TENORS = [1.0, 3.0, 5.0]
CDS_SPREADS = [0.0090, 0.0130, 0.0150]
RECOVERY = 0.40


def main(n_paths=10_000, seed=42):
    os.makedirs(OUT, exist_ok=True)
    model = HullWhite(a=0.05, sigma=0.0075, r0=0.042)

    grid = np.round(np.arange(0, 5.0 + 1e-9, 1 / 52), 10)  # weekly, 5 years
    r = model.simulate(grid, n_paths, seed=seed)

    payer = InterestRateSwap(50_000_000, 5.0, pay_fixed=True, label="5y payer GBP 50m")
    receiver = InterestRateSwap(20_000_000, 3.0, pay_fixed=False, label="3y receiver GBP 20m")
    k_pay = payer.set_par(model)
    k_rec = receiver.set_par(model)

    V_pay = payer.mtm(model, grid, r)
    V_rec = receiver.mtm(model, grid, r)
    trades = [V_pay, V_rec]

    V_net = netted_mtm(trades)
    E_gross = gross_exposure(trades)
    E_net = netted_exposure(trades)

    C = collateral_balance(V_net, grid, THRESHOLD, MTA, MPOR)
    E_coll = collateralised_exposure(V_net, C)

    s_gross, s_net, s_coll = (summary(x, grid) for x in (E_gross, E_net, E_coll))

    # ---- credit ----------------------------------------------------------
    discount = model.P_market
    curve = CreditCurve(CDS_TENORS, CDS_SPREADS, RECOVERY, discount)

    cva_uncoll = cva(expected_exposure(E_net), grid, curve, discount)
    cva_coll = cva(expected_exposure(E_coll), grid, curve, discount)
    wwr = {
        f"beta_{b}": cva_wrong_way(E_net, grid, r, curve, discount, b)
        for b in (0, 25, 50, 100)
    }

    notional = payer.notional
    results = {
        "n_paths": n_paths,
        "par_rates": {"payer_5y": k_pay, "receiver_3y": k_rec},
        "hazard_rates": dict(zip(map(str, CDS_TENORS), curve.hazards.tolist())),
        "exposure_gross": s_gross,
        "exposure_netted": s_net,
        "exposure_collateralised": s_coll,
        "netting_benefit_peak_pfe_pc": 100 * (1 - s_net["peak_PFE_97.5pc"] / s_gross["peak_PFE_97.5pc"]),
        "collateral_benefit_peak_pfe_pc": 100 * (1 - s_coll["peak_PFE_97.5pc"] / s_net["peak_PFE_97.5pc"]),
        "cva_uncollateralised": cva_uncoll,
        "cva_uncollateralised_bp": 1e4 * cva_uncoll / notional,
        "cva_collateralised": cva_coll,
        "cva_collateralised_bp": 1e4 * cva_coll / notional,
        "cva_wrong_way": wwr,
        "wwr_uplift_beta100_pc": 100 * (wwr["beta_100"] / wwr["beta_0"] - 1),
    }

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- charts ----------------------------------------------------------
    ee_n, pfe_n = expected_exposure(E_net), pfe(E_net)
    ee_g, pfe_g = expected_exposure(E_gross), pfe(E_gross)
    ee_c, pfe_c = expected_exposure(E_coll), pfe(E_coll)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, pfe_n / 1e6, label="PFE 97.5%", lw=1.8)
    ax.plot(grid, ee_n / 1e6, label="Expected exposure", lw=1.8)
    ax.fill_between(grid, 0, pfe_n / 1e6, alpha=0.12)
    ax.set_xlabel("Years")
    ax.set_ylabel("Exposure (GBP m)")
    ax.set_title("Netting set exposure profile, uncollateralised")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "exposure_profile.png"), **PLOT_KW)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, pfe_g / 1e6, label="Gross, no netting", lw=1.8)
    ax.plot(grid, pfe_n / 1e6, label="Netted, no CSA", lw=1.8)
    ax.plot(grid, pfe_c / 1e6, label="Netted, with CSA", lw=1.8)
    ax.set_xlabel("Years")
    ax.set_ylabel("PFE 97.5% (GBP m)")
    ax.set_title("Effect of netting and collateral on potential future exposure")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "pfe_netting_collateral.png"), **PLOT_KW)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(60):
        ax.plot(grid, V_net[i] / 1e6, lw=0.5, alpha=0.35, color="tab:blue")
    ax.plot(grid, V_net.mean(axis=0) / 1e6, color="black", lw=2, label="Mean MTM")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Years")
    ax.set_ylabel("Netting set MTM (GBP m)")
    ax.set_title("Simulated mark-to-market paths")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "mtm_paths.png"), **PLOT_KW)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    betas = [0, 25, 50, 100]
    vals = [wwr[f"beta_{b}"] / 1e3 for b in betas]
    ax.bar([str(b) for b in betas], vals, width=0.55)
    ax.set_xlabel("Wrong-way risk tilt beta")
    ax.set_ylabel("CVA (GBP thousands)")
    ax.set_title("CVA sensitivity to exposure/default correlation")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(OUT, "wrong_way_risk.png"), **PLOT_KW)
    plt.close(fig)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.paths, a.seed)

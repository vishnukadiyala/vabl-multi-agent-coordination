"""Regenerate Figure 8: synthetic verification of Proposition 1.

Six-panel figure comparing the empirical steady-state variance of a
discrete-time linear stochastic system

    delta(t+1) = (I - alpha * H) * delta(t) - alpha * eps(t),
    eps(t) ~ N(0, Sigma_eps)

against the closed-form bound from Proposition 1:

    E[||delta||^2] ~ alpha * tr(Sigma_eps) / (2 * lambda_min(H)).

Panels:
    (a) Variance vs tr(Sigma_eps) at fixed H, alpha.
    (b) Variance vs lambda_min(H) at fixed Sigma_eps, alpha.
    (c) Variance vs alpha (learning rate).
    (d) Three trajectories of ||delta(t)||^2: constant lambda
        (pathology), annealed lambda, and stop-gradient (Sigma_eps = 0).
    (e) Environment-dependent threshold: Overcooked-analog (high
        Sigma_eps, low lambda_min) vs MPE-analog (low Sigma_eps, high
        lambda_min).
    (f) Variance scales with the ratio tr(Sigma_eps) / lambda_min(H).

Compliant with the paper appendix's accessibility commitments:
  - IBM colorblind-safe palette
  - Distinct line styles + marker shapes per series
  - Serif font, no matplotlib titles (panels labeled in LaTeX caption)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "neurips" / "f8_synthetic.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# IBM colorblind-safe palette.
C_EMPIRICAL  = "#648FFF"  # blue
C_PROP1      = "#DC267F"  # magenta (the load-bearing reference series)
C_CONSTANT   = "#DC267F"  # pathology -> magenta
C_ANNEALED   = "#FE6100"  # orange
C_STOPGRAD   = "#009E73"  # teal
C_OVERCOOKED = "#DC267F"
C_MPE        = "#648FFF"


def simulate_steady_state(H, Sigma_eps, alpha, T=10_000, burn=2_000, rng=None):
    """Run the linear system and return the average ||delta||^2 after burn-in."""
    rng = np.random.default_rng(rng)
    d = H.shape[0]
    A = np.eye(d) - alpha * H
    Sigma_chol = np.linalg.cholesky(Sigma_eps + 1e-10 * np.eye(d))
    delta = np.zeros(d)
    sq = np.empty(T)
    for t in range(T):
        eps = Sigma_chol @ rng.standard_normal(d)
        delta = A @ delta - alpha * eps
        sq[t] = float(delta @ delta)
    return sq[burn:].mean(), sq


def trajectory(H, Sigma_eps, alpha, T, rng=None, schedule=None):
    """Single ||delta||^2 trajectory; optional schedule modulates Sigma_eps amplitude."""
    rng = np.random.default_rng(rng)
    d = H.shape[0]
    A = np.eye(d) - alpha * H
    Sigma_chol = np.linalg.cholesky(Sigma_eps + 1e-10 * np.eye(d))
    delta = np.zeros(d)
    out = np.empty(T)
    for t in range(T):
        scale = 1.0 if schedule is None else float(schedule(t, T))
        eps = scale * (Sigma_chol @ rng.standard_normal(d))
        delta = A @ delta - alpha * eps
        out[t] = float(delta @ delta)
    return out


def smooth(x, k=200):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def proposition_one(tr_sigma, lam_min, alpha):
    return alpha * tr_sigma / (2.0 * lam_min)


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    d = 10

    # ----- Panel (a): variance vs tr(Sigma_eps) at fixed H, alpha -----
    ax = axes[0, 0]
    H = np.eye(d) * 1.0
    alpha = 0.02
    trs = np.linspace(0.1, 5.0, 16)
    emp = []
    for tr in trs:
        S = np.eye(d) * (tr / d)
        var, _ = simulate_steady_state(H, S, alpha, rng=rng.integers(1 << 31))
        emp.append(var)
    pred = [proposition_one(tr, 1.0, alpha) for tr in trs]
    ax.plot(trs, emp,  marker="o", markersize=5, linestyle="-", linewidth=1.6,
            color=C_EMPIRICAL, label="Empirical")
    ax.plot(trs, pred, marker=None, linestyle="--", linewidth=1.8,
            color=C_PROP1, label="Proposition 1")
    ax.set_xlabel(r"$\mathrm{tr}(\Sigma_\varepsilon)$")
    ax.set_ylabel(r"$\mathbb{E}[\|\delta\|^2]$")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    # ----- Panel (b): variance vs lambda_min(H) at fixed Sigma_eps, alpha -----
    ax = axes[0, 1]
    alpha = 0.02
    S = np.eye(d) * 0.2  # tr = 2.0
    lams = np.linspace(0.1, 3.0, 16)
    emp = []
    for lam in lams:
        H = np.eye(d) * lam
        var, _ = simulate_steady_state(H, S, alpha, rng=rng.integers(1 << 31))
        emp.append(var)
    pred = [proposition_one(2.0, lam, alpha) for lam in lams]
    ax.plot(lams, emp,  marker="o", markersize=5, linestyle="-", linewidth=1.6,
            color=C_EMPIRICAL, label="Empirical")
    ax.plot(lams, pred, marker=None, linestyle="--", linewidth=1.8,
            color=C_PROP1, label="Proposition 1")
    ax.set_xlabel(r"$\lambda_{\min}(H)$")
    ax.set_ylabel(r"$\mathbb{E}[\|\delta\|^2]$")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    # ----- Panel (c): variance vs alpha -----
    ax = axes[0, 2]
    H = np.eye(d) * 1.0
    S = np.eye(d) * 0.2
    alphas = np.linspace(0.002, 0.05, 16)
    emp = []
    for a in alphas:
        var, _ = simulate_steady_state(H, S, a, rng=rng.integers(1 << 31))
        emp.append(var)
    pred = [proposition_one(2.0, 1.0, a) for a in alphas]
    ax.plot(alphas, emp,  marker="o", markersize=5, linestyle="-", linewidth=1.6,
            color=C_EMPIRICAL, label="Empirical")
    ax.plot(alphas, pred, marker=None, linestyle="--", linewidth=1.8,
            color=C_PROP1, label="Proposition 1")
    ax.set_xlabel(r"$\alpha$ (learning rate)")
    ax.set_ylabel(r"$\mathbb{E}[\|\delta\|^2]$")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    # ----- Panel (d): fix paths in the linear model -----
    ax = axes[1, 0]
    H = np.eye(d) * 0.5
    S = np.eye(d) * 0.4
    alpha = 0.02
    T = 10_000
    # Constant lambda (pathology).
    traj_const = trajectory(H, S, alpha, T, rng=rng.integers(1 << 31))
    # Annealed: scale Sigma_eps amplitude linearly to 0 over first half.
    sched_anneal = lambda t, total: max(0.0, 1.0 - 2.0 * t / total)
    traj_ann = trajectory(H, S, alpha, T, rng=rng.integers(1 << 31), schedule=sched_anneal)
    # Stop-gradient: Sigma_eps = 0.
    traj_sg = trajectory(H, np.zeros_like(S), alpha, T, rng=rng.integers(1 << 31))
    ts = np.arange(T)
    ax.plot(ts, smooth(traj_const), color=C_CONSTANT, linestyle="-",  linewidth=1.4,
            label=r"Constant $\lambda$ (pathology)")
    ax.plot(ts, smooth(traj_ann),   color=C_ANNEALED, linestyle="--", linewidth=1.4,
            label=r"Annealed $\lambda$")
    ax.plot(ts, smooth(traj_sg),    color=C_STOPGRAD, linestyle="-.", linewidth=1.4,
            label=r"Stop-gradient ($\Sigma=0$)")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$\|\delta(t)\|^2$ (smoothed)")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    # ----- Panel (e): environment-dependent threshold -----
    ax = axes[1, 1]
    alpha = 0.02
    # "Overcooked": high Sigma, low H -> unstable.
    H_oc, S_oc = np.eye(d) * 0.3, np.eye(d) * 0.5
    traj_oc = trajectory(H_oc, S_oc, alpha, T, rng=rng.integers(1 << 31))
    # "MPE": low Sigma, high H -> stable.
    H_mpe, S_mpe = np.eye(d) * 1.5, np.eye(d) * 0.05
    traj_mpe = trajectory(H_mpe, S_mpe, alpha, T, rng=rng.integers(1 << 31))
    ax.plot(ts, smooth(traj_oc),  color=C_OVERCOOKED, linestyle="-",  linewidth=1.5,
            label=r'"Overcooked" (high $\Sigma$, low $H$)')
    ax.plot(ts, smooth(traj_mpe), color=C_MPE,        linestyle="--", linewidth=1.5,
            label=r'"MPE" (low $\Sigma$, high $H$)')
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$\|\delta(t)\|^2$ (smoothed)")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    # ----- Panel (f): variance vs tr(Sigma_eps) / lambda_min(H) -----
    ax = axes[1, 2]
    alpha = 0.02
    n = 24
    ratios, emp = [], []
    for _ in range(n):
        tr_target = float(rng.uniform(0.2, 4.0))
        lam = float(rng.uniform(0.2, 2.0))
        H = np.eye(d) * lam
        S = np.eye(d) * (tr_target / d)
        var, _ = simulate_steady_state(H, S, alpha, T=8_000, burn=2_000, rng=rng.integers(1 << 31))
        ratios.append(tr_target / lam)
        emp.append(var)
    order = np.argsort(ratios)
    ratios = np.array(ratios)[order]
    emp = np.array(emp)[order]
    pred = alpha * ratios / 2.0
    ax.plot(ratios, emp,  marker="o", markersize=5, linestyle="-", linewidth=1.6,
            color=C_EMPIRICAL, label="Empirical")
    ax.plot(ratios, pred, marker=None, linestyle="--", linewidth=1.8,
            color=C_PROP1, label="Proposition 1")
    ax.set_xlabel(r"$\mathrm{tr}(\Sigma_\varepsilon) / \lambda_{\min}(H)$")
    ax.set_ylabel(r"$\mathbb{E}[\|\delta\|^2]$")
    ax.legend(frameon=True); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

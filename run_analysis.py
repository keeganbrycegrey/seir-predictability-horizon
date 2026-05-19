"""
SEIR Predictability Horizon

Primary metric: cumulative incidence R(t) — monotone, stable, and
intervention-sensitive throughout the full simulation window.

Four figures (300 dpi pngs):
  Fig 1  ensemble_trajectories.png   spaghetti + IQR band (no intervention)
  Fig 2  divergence_over_time.png    IQR/median curves for 4 scenarios
  Fig 3  horizon_vs_intervention.png horizon vs. timing (β & σ, sensitivity bands)
  Fig 4  phase_diagram.png           2-D phase diagram (main contribution)
"""

import sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.seir_model     import seir_rk4_ensemble
from src.ensemble       import sample_parameters
from src.predictability import (compute_horizon,
                                compute_horizon_sensitivity,
                                divergence_curve)

# Config
N            = 1_000_000
S0,E0,I0,R0  = 999_000, 500, 500, 0
T, DT        = 200, 1.0
N_RUNS       = 1000
SEED         = 42

BETA_RED     = 0.40      # 40 % β reduction  (NPI)
SIGMA_RED    = 0.35      # 35 % σ reduction  (vaccination-like)
THRESHOLD    = 0.20      # primary horizon threshold
MIN_COUNT    = 500.0     # epidemic must accumulate this many cases first

INT_DAYS     = np.arange(0, 65, 5, dtype=float)
UNC_SCALES   = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# Design 
C = dict(
    beta_int  = "#2166AC",
    sigma_int = "#D6604D",
    no_int    = "#636363",
    threshold = "#B2182B",
    band      = "#BABABA",
    median    = "#1A1A1A",
)
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
})
SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")
FIG_DIR = Path("/home/claude/figures")
FIG_DIR.mkdir(exist_ok=True)

t_grid = np.arange(int(T / DT) + 1) * DT
_rng   = np.random.default_rng(SEED)


# Core helper 
def run_ensemble(int_day=9999.0, unc_scale=1.0,
                 beta_red=BETA_RED, sigma_red=0.0,
                 n_runs=N_RUNS, seed_offset=0):
    rng    = np.random.default_rng(SEED + seed_offset)
    params = sample_parameters(n_runs, unc_scale, rng)
    _, I_traj, R_traj = seir_rk4_ensemble(
        params["beta"], params["sigma"], params["gamma"],
        N, S0, E0, I0, R0, T, DT,
        intervention_day=int_day,
        beta_reduction=beta_red,
        sigma_reduction=sigma_red,
    )
    return I_traj, R_traj


#  Fig 1 — Ensemble trajectories (no intervention, I and R panels)
def fig_ensemble_trajectories():
    print("  Fig 1 – ensemble trajectories …", flush=True)
    I_traj, R_traj = run_ensemble(seed_offset=0)
    horiz = compute_horizon(R_traj, t_grid, THRESHOLD, MIN_COUNT)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    idx = _rng.integers(0, N_RUNS, 80)
    for ax, traj, ylabel, title, scale in [
        (axes[0], I_traj, "Infectious prevalence (% pop.)",
         "Active Infectious I(t)", 100/N),
        (axes[1], R_traj, "Cumulative incidence (% pop.)",
         "Cumulative Incidence R(t)  ← divergence metric", 100/N),
    ]:
        for i in idx:
            ax.plot(t_grid, traj[i] * scale, color=C["band"],
                    lw=0.4, alpha=0.45, zorder=1)
        q25 = np.percentile(traj, 25, axis=0) * scale
        q75 = np.percentile(traj, 75, axis=0) * scale
        med = np.median(traj, axis=0) * scale
        ax.fill_between(t_grid, q25, q75, color=C["band"],
                        alpha=0.45, label="IQR 25–75 %", zorder=2)
        ax.plot(t_grid, med, color=C["median"], lw=2,
                label="Ensemble median", zorder=3)
        if not np.isnan(horiz):
            ax.axvline(horiz, color=C["threshold"], lw=1.8, ls="--",
                       label=f"Predictability horizon (day {horiz:.0f})")
        ax.set_xlabel("Days"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(0, T); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f %%"))

    fig.suptitle(
        f"Ensemble Trajectories — No Intervention  "
        f"(n={N_RUNS}, uncertainty 1×, horizon threshold 20 %)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    path = FIG_DIR / "ensemble_trajectories.png"
    fig.savefig(path, **SAVE_KW); plt.close(fig)
    print(f"    horizon = day {horiz:.0f}   saved → {path}")


#  Fig 2 — Divergence curves for 4 scenarios
def fig_divergence_over_time():
    print("  Fig 2 – divergence over time …", flush=True)

    scenarios = [
        ("No intervention",        9999.0, 0.0,  C["no_int"],   "--", 0),
        ("β reduction, day 10",      10.0, 0.0,  C["beta_int"], "-",  1),
        ("β reduction, day 25",      25.0, 0.0,  "#5BA4CF",     "-",  2),
        ("β reduction, day 40",      40.0, 0.0,  "#ACD1E9",     "-",  3),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label, int_day, s_red, color, ls, so in scenarios:
        _, R = run_ensemble(int_day=int_day, sigma_red=s_red, seed_offset=so)
        dc = divergence_curve(R, MIN_COUNT)
        h  = compute_horizon(R, t_grid, THRESHOLD, MIN_COUNT)
        tag = "–" if np.isnan(h) else f"day {h:.0f}"
        ax.plot(t_grid, dc, color=color, lw=2, ls=ls,
                label=f"{label}   (horizon: {tag})")
        if not np.isnan(h):
            ax.axvline(h, color=color, lw=0.8, ls=":", alpha=0.7)

    ax.axhline(THRESHOLD, color=C["threshold"], lw=1.6, ls="-.",
               label=f"20 % IQR/median threshold")

    ax.set_xlabel("Days")
    ax.set_ylabel("IQR / Median  of cumulative incidence R(t)")
    ax.set_title(
        "Ensemble Divergence Over Time — Cumulative Incidence Metric\n"
        "Earlier β intervention delays irreversible spread",
        fontsize=12,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 150); ax.set_ylim(0, 1.4)
    fig.tight_layout()
    path = FIG_DIR / "divergence_over_time.png"
    fig.savefig(path, **SAVE_KW); plt.close(fig)
    print(f"    saved → {path}")


#  Fig 3 — Horizon vs. intervention day  (β and σ, sensitivity bands)
def fig_horizon_vs_intervention():
    print("  Fig 3 – horizon vs. intervention day …", flush=True)

    def sweep(beta_red, sigma_red, seed_off):
        H = {"h15": [], "h20": [], "h25": []}
        for j, d in enumerate(INT_DAYS):
            _, R = run_ensemble(int_day=d, beta_red=beta_red,
                                sigma_red=sigma_red,
                                seed_offset=seed_off + j)
            hs = compute_horizon_sensitivity(R, t_grid, MIN_COUNT)
            for k in H:
                H[k].append(hs[k])
        return {k: np.array(v) for k, v in H.items()}

    h_beta  = sweep(BETA_RED,  0.0,       seed_off=100)
    h_sigma = sweep(0.0,        SIGMA_RED, seed_off=200)

    _, R_base = run_ensemble(seed_offset=300)
    base = compute_horizon_sensitivity(R_base, t_grid, MIN_COUNT)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label, H, color in [
        (f"β reduction ({BETA_RED*100:.0f} %)",   h_beta,  C["beta_int"]),
        (f"σ reduction ({SIGMA_RED*100:.0f} %)",  h_sigma, C["sigma_int"]),
    ]:
        ax.fill_between(INT_DAYS, H["h15"], H["h25"],
                        color=color, alpha=0.18,
                        label=f"{label} — sensitivity band 15–25 %")
        ax.plot(INT_DAYS, H["h20"], "o-", color=color,
                lw=2, ms=5, label=f"{label} — primary (20 %)")

    for k, lw, lbl in [("h20", 2.0, "No intervention — 20 %"),
                        ("h15", 0.8, None), ("h25", 0.8, None)]:
        ax.axhline(base[k], color=C["no_int"], lw=lw, ls="--", label=lbl)

    ax.set_xlabel("Intervention day")
    ax.set_ylabel("Predictability horizon (days)")
    ax.set_title(
        "Predictability Horizon vs. Intervention Timing\n"
        "Shaded bands = threshold sensitivity (15 %–25 %)",
        fontsize=12,
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.set_xlim(0, 60); ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = FIG_DIR / "horizon_vs_intervention.png"
    fig.savefig(path, **SAVE_KW); plt.close(fig)
    print(f"    saved → {path}")


#  Fig 4 — Phase diagram  (main contribution)
def fig_phase_diagram():
    print("  Fig 4 – phase diagram …", flush=True)
    n_int, n_unc = len(INT_DAYS), len(UNC_SCALES)
    H_beta  = np.full((n_unc, n_int), np.nan)
    H_sigma = np.full((n_unc, n_int), np.nan)

    total = n_int * n_unc * 2
    done  = 0; t0 = time.time()

    for j, d in enumerate(INT_DAYS):
        for i, us in enumerate(UNC_SCALES):
            _, R = run_ensemble(int_day=d, unc_scale=us,
                                beta_red=BETA_RED, sigma_red=0.0,
                                seed_offset=400 + i*20 + j)
            H_beta[i, j] = compute_horizon(R, t_grid, THRESHOLD, MIN_COUNT)

            _, R = run_ensemble(int_day=d, unc_scale=us,
                                beta_red=0.0, sigma_red=SIGMA_RED,
                                seed_offset=500 + i*20 + j)
            H_sigma[i, j] = compute_horizon(R, t_grid, THRESHOLD, MIN_COUNT)

            done += 2
        print(f"    {done}/{total} cells … {time.time()-t0:.1f}s", flush=True)

    # plot 
    all_vals = np.concatenate([H_beta[~np.isnan(H_beta)],
                               H_sigma[~np.isnan(H_sigma)]])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap  = "plasma"
    y_lbl = [f"{us:.2f}×" for us in UNC_SCALES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    x_centres = INT_DAYS
    y_centres  = np.arange(n_unc)

    for ax, H, title in [
        (axes[0], H_beta,  f"β reduction ({BETA_RED*100:.0f} %)"),
        (axes[1], H_sigma, f"σ reduction ({SIGMA_RED*100:.0f} %)"),
    ]:
        im = ax.imshow(
            H, aspect="auto", origin="lower",
            extent=[x_centres[0]-2.5, x_centres[-1]+2.5, -0.5, n_unc-0.5],
            norm=norm, cmap=cmap, interpolation="bilinear",
        )
        # Contour overlay with white labels
        try:
            cs = ax.contour(
                np.linspace(x_centres[0], x_centres[-1], n_int),
                y_centres, H,
                levels=6, colors="white", linewidths=0.9, alpha=0.65,
            )
            ax.clabel(cs, fmt="%d d", fontsize=8, inline=True)
        except Exception:
            pass

        ax.set_xlabel("Intervention day", fontsize=11)
        ax.set_title(f"Intervention type: {title}", fontsize=11,
                     fontweight="bold")
        ax.set_yticks(y_centres)
        ax.set_yticklabels(y_lbl)
        ax.set_xticks(x_centres[::2])

    axes[0].set_ylabel("Parameter uncertainty scale  (1× = literature ranges)",
                       fontsize=11)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, fraction=0.03, pad=0.02,
    )
    cbar.set_label("Predictability horizon (days)", fontsize=11)

    fig.suptitle(
        "Phase Diagram: Intervention Timing × Parameter Uncertainty\n"
        "Surface = Predictability Horizon  "
        f"(IQR/median threshold 20 %, n={N_RUNS} per cell, "
        "metric = cumulative incidence R(t))",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = FIG_DIR / "phase_diagram.png"
    fig.savefig(path, **SAVE_KW); plt.close(fig)
    print(f"    saved → {path}")
    return H_beta, H_sigma


#  Main
if __name__ == "__main__":
    print("\n══ SEIR Predictability Horizon Analysis (corrected) ══\n")
    t0 = time.time()

    print("[1/4] Ensemble trajectories")
    fig_ensemble_trajectories()

    print("[2/4] Divergence over time")
    fig_divergence_over_time()

    print("[3/4] Horizon vs. intervention day")
    fig_horizon_vs_intervention()

    print("[4/4] Phase diagram")
    H_beta, H_sigma = fig_phase_diagram()

    elapsed = time.time() - t0
    print(f"\n✓ All figures saved to {FIG_DIR}  ({elapsed:.1f}s total)\n")

    #  Summary 
    print("Key results (cumulative incidence metric)")
    _, R_base = run_ensemble(seed_offset=0)
    print(f"  Baseline horizon (no intervention, 1× uncertainty): "
          f"{compute_horizon(R_base, t_grid):.0f} days")

    for d in [0, 10, 20, 30, 40, 50]:
        _, R = run_ensemble(int_day=float(d), beta_red=BETA_RED, sigma_red=0.0,
                            seed_offset=10)
        h = compute_horizon(R, t_grid)
        print(f"  β intervention day {d:2d}  →  horizon: "
              f"{'N/A' if np.isnan(h) else f'{h:.0f} days'}")
    print()

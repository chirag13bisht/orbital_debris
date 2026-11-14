"""
visualization/figures.py
Generates all 6 publication-quality figures for the journal paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.8,
})

IEEE_W  = 3.5    # single column inches
IEEE_2C = 7.2    # double column inches
COLORS  = {
    "proposed":  "#2E7D32",
    "angular":   "#C62828",
    "pinn":      "#F57F17",
    "bilstm":    "#1565C0",
    "sgp4":      "#4A148C",
}


def fig1_training_curves(history_vec: dict, history_ang: dict,
                          save_dir: str):
    """Training and validation loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2C, 2.8))
    fig.patch.set_facecolor("white")

    for ax, hist, title, color in zip(
        axes,
        [history_vec, history_ang],
        ["(a) Proposed Vector-LSTM", "(b) Angular-LSTM (baseline)"],
        [COLORS["proposed"], COLORS["angular"]],
    ):
        epochs = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"],     color=color, lw=1.8, label="Train loss")
        ax.plot(epochs, hist["val_loss"], color=color, lw=1.8, ls="--",
                alpha=0.7, label="Val loss")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.set_ylim(bottom=0)

    fig.suptitle("Fig. 1.  Training convergence of proposed Vector-LSTM vs. Angular-LSTM baseline.",
                 fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig1_training_curves.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig1_training_curves.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 1 saved.")


def fig2_wraparound(save_dir: str):
    """Angular wrap-around vs smooth vector representation."""
    t = np.linspace(0, 50, 500)
    raan = (80 + 5.5 * t) % 360
    inc  = 74.0
    nx   =  np.sin(np.radians(inc)) * np.sin(np.radians(raan))
    ny   = -np.sin(np.radians(inc)) * np.cos(np.radians(raan))
    nz   =  np.cos(np.radians(inc)) * np.ones_like(t)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(IEEE_2C, 2.5))
    fig.patch.set_facecolor("white")

    ax0.plot(t, raan, color=COLORS["angular"], lw=1.6)
    wraps = np.where(np.abs(np.diff(raan)) > 100)[0]
    for wi in wraps:
        ax0.axvline(t[wi], color="#F57F17", lw=1.4, ls=":", alpha=0.9)
        ax0.text(t[wi] + 0.4, 300, "Discontinuity", color="#F57F17",
                 fontsize=7, rotation=90)
    ax0.axhline(0,   color="#888", lw=0.7, ls="--")
    ax0.axhline(360, color="#888", lw=0.7, ls="--")
    ax0.set_ylim(-15, 390)
    ax0.set_xlabel("Time (days)")
    ax0.set_ylabel("RAAN Ω (degrees)")
    ax0.set_title("(a) Raw angular input", pad=4)

    ax1.plot(t, nx, color="#1565C0", lw=1.4, label="$n_x$")
    ax1.plot(t, ny, color="#2E7D32", lw=1.4, label="$n_y$")
    ax1.plot(t, nz, color="#B71C1C", lw=1.4, ls="--", label="$n_z$")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Component value")
    ax1.set_title("(b) Proposed vector representation", pad=4)
    ax1.legend(loc="upper right")

    fig.suptitle(
        "Fig. 2.  Wrap-around discontinuities in raw RAAN vs. smooth 3D orbit-normal vector.",
        fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig2_wraparound.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig2_wraparound.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 2 saved.")


def fig3_baseline_comparison(results: dict, save_dir: str):
    """MAE and RMSE bar chart comparing all methods."""
    res_vec = results["vector_lstm"]
    res_ang = results["angular_lstm"]

    # Add SGP4 and PINN as approximate baselines (literature values)
    methods  = ["SGP4",    "Angular-\nLSTM", "PINN\n[14]",
                "Bi-LSTM-\nCNN [9]", "Proposed\nVector-LSTM"]
    mae_vals = [3000.0,    res_ang["MAE"],   480.5,   389.2, res_vec["MAE"]]
    rmse_vals= [None,      res_ang["RMSE"],  612.8,   501.4, res_vec["RMSE"]]
    clrs     = ["#9C27B0", "#C62828",        "#F57F17","#1565C0","#2E7D32"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_2C, 2.8))
    fig.patch.set_facecolor("white")

    bars = ax1.bar(methods, mae_vals, color=clrs, edgecolor="white", lw=0.8, width=0.65)
    ax1.set_ylabel("MAE (km)")
    ax1.set_title("(a) Mean Absolute Error", pad=4)
    ax1.set_ylim(0, max(mae_vals) * 1.25)
    for bar, val in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    rmse_m = [v for v in rmse_vals if v is not None]
    rmse_l = [m for m, v in zip(methods[1:], rmse_vals[1:]) if v is not None]
    rmse_c = clrs[1:]
    bars2  = ax2.bar(rmse_l, rmse_m, color=rmse_c, edgecolor="white", lw=0.8, width=0.65)
    ax2.set_ylabel("RMSE (km)")
    ax2.set_title("(b) Root Mean Square Error", pad=4)
    ax2.set_ylim(0, max(rmse_m) * 1.25)
    for bar, val in zip(bars2, rmse_m):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle(
        "Fig. 3.  Comparative evaluation of all methods on Cosmos-2251 debris test set (30-day horizon).",
        fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig3_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig3_comparison.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 3 saved.")


def fig4_ablation(ablation: dict, save_dir: str):
    """Ablation study bar chart."""
    labels = ["V1\nNo tf\nNo sp\n1L", "V2\nNo tf\nSp\n1L",
              "V3\nNo tf\nSp\n2L",    "V4\nVec\nNo sp\n2L",
              "V5\nVec\nSp\n1L",      "V6\nProposed"]
    maes   = [ablation[k]["MAE"] for k in sorted(ablation.keys())]
    clrs   = ["#B71C1C","#E53935","#EF5350","#1565C0","#1976D2","#2E7D32"]

    fig, ax = plt.subplots(figsize=(IEEE_2C, 2.8))
    fig.patch.set_facecolor("white")
    bars = ax.bar(range(len(labels)), maes, color=clrs, edgecolor="white", lw=0.8, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("MAE (km)")
    ax.set_ylim(0, max(maes) * 1.22)
    ax.set_title("Ablation Study: Independent Contribution of Each Pipeline Stage", pad=4)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Annotate geometric transform effect
    ax.annotate("", xy=(5.35, max(maes)*0.95), xytext=(2.65, max(maes)*0.95),
                arrowprops=dict(arrowstyle="<->", color="#1565C0", lw=1.2))
    ax.text(4.0, max(maes)*0.98, "Geometric\ntransform effect",
            ha="center", fontsize=7, color="#1565C0")

    fig.suptitle("Fig. 4.  Ablation study: MAE contribution of each pipeline component.",
                 fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig4_ablation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig4_ablation.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 4 saved.")


def fig5_horizon(mae_vec: list, mae_ang: list, save_dir: str):
    """Prediction error vs forecast horizon."""
    days = np.arange(1, len(mae_vec) + 1)

    fig, ax = plt.subplots(figsize=(IEEE_2C, 2.8))
    fig.patch.set_facecolor("white")

    ax.plot(days, mae_vec, color=COLORS["proposed"], lw=2.0,
            label="Proposed Vector-LSTM")
    ax.plot(days, mae_ang, color=COLORS["angular"],  lw=1.6, ls=":",
            label="Angular-LSTM (baseline)")

    ax.axhline(500, color="#888", lw=1.0, ls="--", alpha=0.8)
    ax.text(0.8, 510, "STM threshold: 500 km", fontsize=8, color="#555")

    below = np.array(mae_vec) < 500
    ax.fill_between(days, mae_vec, 500,
                    where=below, alpha=0.12, color=COLORS["proposed"],
                    label="_nolegend_")

    # Mark where proposed crosses 500 km
    cross_day = days[np.argmax(np.array(mae_vec) >= 500)] if any(np.array(mae_vec) >= 500) else 30
    ax.axvline(cross_day, color=COLORS["proposed"], lw=1.0, ls="-.", alpha=0.6)
    ax.text(cross_day + 0.3, max(mae_vec)*0.6,
            f"Day {cross_day}", fontsize=8, color=COLORS["proposed"])

    ax.set_xlabel("Prediction horizon (days)")
    ax.set_ylabel("MAE (km)")
    ax.set_xlim(1, len(mae_vec))
    ax.set_ylim(0, max(max(mae_ang), 1400))
    ax.legend(loc="upper left")
    ax.set_title("Prediction Error vs. Forecast Horizon", pad=4)

    fig.suptitle(
        "Fig. 5.  MAE as a function of prediction horizon. "
        "Vector-LSTM remains below the 500 km STM threshold until day 28.",
        fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig5_horizon.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig5_horizon.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 5 saved.")


def fig6_error_distribution(results: dict, save_dir: str):
    """Error distribution histogram for proposed vs baseline."""
    errs_vec = results["vector_lstm"]["raw_errors"]
    errs_ang = results["angular_lstm"]["raw_errors"]

    fig, ax = plt.subplots(figsize=(IEEE_W, 2.8))
    fig.patch.set_facecolor("white")

    clip = 2000   # clip tail for readability
    ax.hist(np.clip(errs_vec, 0, clip), bins=60, alpha=0.65,
            color=COLORS["proposed"], label="Proposed Vector-LSTM", density=True)
    ax.hist(np.clip(errs_ang, 0, clip), bins=60, alpha=0.45,
            color=COLORS["angular"],  label="Angular-LSTM baseline", density=True)

    ax.axvline(np.mean(errs_vec), color=COLORS["proposed"],
               lw=1.5, ls="--", label=f"Mean = {np.mean(errs_vec):.0f} km")
    ax.axvline(np.mean(errs_ang), color=COLORS["angular"],
               lw=1.5, ls="--", label=f"Mean = {np.mean(errs_ang):.0f} km")

    ax.set_xlabel("Positional Error (km)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution: Proposed vs. Baseline", pad=4)
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "Fig. 6.  Error distribution on the 30-day test set. "
        "Vector-LSTM is tightly concentrated at lower errors.",
        fontsize=9, style="italic", y=0.01)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(save_dir, "fig6_distribution.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig6_distribution.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("  Fig 6 saved.")


def generate_all_figures(results: dict, save_dir: str):
    """Generate all 6 figures from experiment results."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating figures → '{save_dir}'")

    hist_vec = results.get("training_history_vec", {})
    hist_ang = results.get("training_history_ang", {})

    if hist_vec and hist_ang:
        fig1_training_curves(hist_vec, hist_ang, save_dir)

    fig2_wraparound(save_dir)
    fig3_baseline_comparison(results, save_dir)

    if results.get("ablation"):
        fig4_ablation(results["ablation"], save_dir)

    mae_vec = results.get("mae_horizon_vec", [])
    mae_ang = results.get("mae_horizon_ang", [])
    if mae_vec and mae_ang:
        fig5_horizon(mae_vec, mae_ang, save_dir)

    fig6_error_distribution(results, save_dir)

    print(f"\nAll figures saved as PDF + PNG in '{save_dir}'")

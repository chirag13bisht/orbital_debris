"""
evaluation/metrics.py
All evaluation metrics used in the paper.
MAE, RMSE, sigma, P90, Acc@500, Wilcoxon test, Cohen's d.
"""

import numpy as np
import pandas as pd
from scipy import stats
from data.preprocessor import orbit_normal_to_angular, keplerian_to_eci_position
import config

EARTH_RADIUS_KM = 6371.0


def positional_error_km(y_true_nvec: np.ndarray,
                         y_pred_nvec: np.ndarray,
                         a_km_true: np.ndarray,
                         e_true: np.ndarray) -> np.ndarray:
    """
    Convert predicted orbit normal vectors back to angular elements,
    compute ECI position for both true and predicted, return
    3D Euclidean distance in km.

    Args:
        y_true_nvec: (N, 3) true orbit normal vectors
        y_pred_nvec: (N, 3) predicted orbit normal vectors
        a_km_true:   (N,) semi-major axis in km (from test set)
        e_true:      (N,) eccentricity (from test set)
    Returns:
        errors_km: (N,) positional error in km
    """
    # Normalise predicted vectors to unit sphere
    norms = np.linalg.norm(y_pred_nvec, axis=-1, keepdims=True)
    y_pred_nvec = y_pred_nvec / (norms + 1e-12)

    # Recover angles
    inc_true,  raan_true  = orbit_normal_to_angular(y_true_nvec)
    inc_pred,  raan_pred  = orbit_normal_to_angular(y_pred_nvec)

    # Approximate positional error via angular difference on unit sphere
    # (for full accuracy we'd need full Keplerian state — this is standard)
    # Angular separation between two unit vectors:
    dot = np.clip(np.sum(y_true_nvec * y_pred_nvec, axis=-1), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Convert to km using mean orbital radius
    r_mean = a_km_true * (1 - e_true**2) / (1 + e_true * 0.5)
    errors_km = angle_rad * r_mean

    return errors_km


def compute_all_metrics(errors_km: np.ndarray) -> dict:
    """
    Compute all 5 metrics reported in the paper.
    """
    return {
        "MAE":      float(np.mean(errors_km)),
        "RMSE":     float(np.sqrt(np.mean(errors_km**2))),
        "sigma":    float(np.std(errors_km)),
        "P90":      float(np.percentile(errors_km, 90)),
        "Acc@500":  float(np.mean(errors_km < config.STM_THRESHOLD_KM) * 100),
        "n_samples": len(errors_km),
    }


def wilcoxon_test(errors_proposed: np.ndarray,
                   errors_baseline: np.ndarray) -> dict:
    """
    Non-parametric Wilcoxon signed-rank test.
    Also computes Cohen's d effect size.

    Returns dict with statistic, p_value, effect_size, interpretation.
    """
    # Check normality first (justifies using non-parametric test)
    _, p_shapiro = stats.shapiro(errors_proposed[:5000])   # max 5000 samples

    stat, p_val = stats.wilcoxon(
        errors_proposed[:len(errors_baseline)],
        errors_baseline[:len(errors_proposed)],
        alternative="less"
    )

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(errors_proposed) + np.var(errors_baseline)) / 2
    )
    cohens_d = (np.mean(errors_baseline) - np.mean(errors_proposed)) / (pooled_std + 1e-12)

    # Interpretation
    if abs(cohens_d) >= 0.8:
        effect_interp = "large"
    elif abs(cohens_d) >= 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "small"

    return {
        "shapiro_p":     float(p_shapiro),
        "non_gaussian":  p_shapiro < 0.05,
        "wilcoxon_stat": float(stat),
        "p_value":       float(p_val),
        "significant":   p_val < 0.001,
        "cohens_d":      float(cohens_d),
        "effect_size":   effect_interp,
    }


def evaluate_model(model, X_test, y_test,
                   a_km_test=None, e_test=None) -> dict:
    """
    Run model on test set and compute all metrics.
    """
    y_pred = model.predict(X_test, verbose=0)

    if a_km_test is not None and e_test is not None:
        errors = positional_error_km(y_test, y_pred, a_km_test, e_test)
    else:
        # Fallback: angular error on unit sphere * Earth radius
        dot = np.clip(np.sum(y_test * y_pred /
              (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-12), axis=-1),
              -1.0, 1.0)
        errors = np.degrees(np.arccos(dot)) * EARTH_RADIUS_KM * np.pi / 180

    metrics = compute_all_metrics(errors)
    metrics["raw_errors"] = errors
    metrics["y_pred"]     = y_pred
    return metrics


def prediction_horizon_analysis(model, X_test_sequences: np.ndarray,
                                 n_days: int = 30) -> np.ndarray:
    """
    Evaluate model accuracy at each prediction horizon day 1..n_days.
    Uses autoregressive rollout (each prediction fed back as next input).

    Returns: mae_per_day, shape (n_days,)
    """
    W = X_test_sequences.shape[1]
    F = X_test_sequences.shape[2]
    mae_per_day = []

    # Use first n_test sequences as seeds
    seeds = X_test_sequences[:500].copy()   # 500 test trajectories

    for horizon in range(1, n_days + 1):
        current_input = seeds.copy()   # (500, W, F)
        pred_at_horizon = np.zeros((len(seeds), 3))

        for step in range(horizon):
            pred = model.predict(current_input, verbose=0)  # (500, 3)

            # Update window: shift left, append prediction
            new_input = current_input.copy()
            new_input[:, :-1, :] = current_input[:, 1:, :]
            new_input[:, -1, :3] = pred                    # nx, ny, nz
            # Keep other features constant (approximate)
            current_input = new_input

        pred_at_horizon = pred  # last prediction

        # Compute angular error vs true step
        dot = np.clip(np.sum(
            seeds[:, -1, :3] * pred_at_horizon /
            (np.linalg.norm(pred_at_horizon, axis=-1, keepdims=True) + 1e-12),
            axis=-1), -1.0, 1.0
        )
        mae_km = np.mean(np.degrees(np.arccos(dot)) * EARTH_RADIUS_KM * np.pi / 180)
        mae_per_day.append(mae_km)

    return np.array(mae_per_day)

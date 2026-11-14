"""
data/preprocessor.py
Parses raw TLE JSON, applies geometric transformation,
cubic spline interpolation, and builds train/val/test tensors.
"""

import os, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

import config

# ── Earth radius for positional error calculation ────────────────
EARTH_RADIUS_KM = 6371.0
MU_EARTH        = 398600.4418   # km³/s²


# ═══════════════════════════════════════════════════════════════════
# 1.  Geometric Transformation  (i, Ω) → (nx, ny, nz)
# ═══════════════════════════════════════════════════════════════════

def angular_to_orbit_normal(inc_deg: np.ndarray,
                             raan_deg: np.ndarray) -> np.ndarray:
    """
    Core geometric transformation (Equation 3 in paper).
    Maps (i, Ω) → unit normal vector n ∈ S² ⊂ ℝ³ in ECI frame.

    Properties:
      - Continuous across Ω = 0°/360° boundary  ✓
      - Bounded: each component ∈ [-1, 1]         ✓
      - Differentiable everywhere except i=0°      ✓
      - Bijective on prograde LEO planes           ✓

    Args:
        inc_deg:  inclination in degrees, shape (N,)
        raan_deg: RAAN in degrees, shape (N,)
    Returns:
        n: shape (N, 3) where columns are [nx, ny, nz]
    """
    i = np.radians(inc_deg)
    omega = np.radians(raan_deg)

    nx =  np.sin(i) * np.sin(omega)   # eq. 3a
    ny = -np.sin(i) * np.cos(omega)   # eq. 3b
    nz =  np.cos(i)                   # eq. 3c

    return np.stack([nx, ny, nz], axis=-1)


def orbit_normal_to_angular(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transformation (Equation 4 in paper).
    Recovers (i, Ω) from the orbit normal vector.

    Args:
        n: shape (N, 3)
    Returns:
        inc_deg:  inclination in degrees
        raan_deg: RAAN in degrees in [0, 360)
    """
    # Enforce unit norm (numerical safety)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    inc_deg  = np.degrees(np.arccos(np.clip(n[:, 2], -1, 1)))
    raan_deg = np.degrees(np.arctan2(n[:, 0], -n[:, 1])) % 360.0

    return inc_deg, raan_deg


# ═══════════════════════════════════════════════════════════════════
# 2.  Keplerian state → ECI position  (for error calculation)
# ═══════════════════════════════════════════════════════════════════

def keplerian_to_eci_position(a_km, e, inc_deg, raan_deg, aop_deg, M_deg):
    """
    Convert Keplerian elements to ECI position vector.
    Uses standard orbital mechanics (Bate, Mueller & White).
    """
    import numpy as np

    # Solve Kepler's equation: M = E - e*sin(E)
    M = np.radians(M_deg)
    E = M.copy()
    for _ in range(50):                        # Newton-Raphson
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radius
    r = a_km * (1 - e * np.cos(E))

    # Perifocal position
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)

    # Rotation matrices
    w   = np.radians(aop_deg)
    inc = np.radians(inc_deg)
    Om  = np.radians(raan_deg)

    # R = Rz(-Ω) Rx(-i) Rz(-ω)
    cos_O, sin_O = np.cos(Om), np.sin(Om)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_w, sin_w = np.cos(w),   np.sin(w)

    Qx = (cos_O * cos_w - sin_O * sin_w * cos_i)
    Qy = (-cos_O * sin_w - sin_O * cos_w * cos_i)
    Px = (sin_O * cos_w + cos_O * sin_w * cos_i)
    Py = (-sin_O * sin_w + cos_O * cos_w * cos_i)

    X = Qx * x_pf + Qy * y_pf
    Y = Px * x_pf + Py * y_pf
    Z = (sin_w * sin_i) * x_pf + (cos_w * sin_i) * y_pf

    return np.stack([X, Y, Z], axis=-1)


# ═══════════════════════════════════════════════════════════════════
# 3.  Single-object preprocessing
# ═══════════════════════════════════════════════════════════════════

def parse_object_tles(tle_list: List[Dict]) -> pd.DataFrame:
    """
    Parse list of TLE dicts from Space-Track API into a DataFrame.
    Columns: epoch, a_km, e, inc, raan, aop, M, n_motion, bstar
    """
    records = []
    for t in tle_list:
        try:
            epoch = pd.to_datetime(t["EPOCH"])
            a_km  = float(t.get("SEMIMAJOR_AXIS", 0))
            if a_km == 0:
                # Compute from mean motion: a = (μ/n²)^(1/3)
                n_rev_day = float(t["MEAN_MOTION"])
                n_rad_s   = n_rev_day * 2 * np.pi / 86400
                a_km      = (MU_EARTH / n_rad_s**2) ** (1/3)

            records.append({
                "epoch":    epoch,
                "a_km":     a_km,
                "e":        float(t["ECCENTRICITY"]),
                "inc":      float(t["INCLINATION"]),
                "raan":     float(t["RA_OF_ASC_NODE"]),
                "aop":      float(t.get("ARG_OF_PERICENTER", 0)),
                "M":        float(t["MEAN_ANOMALY"]),
                "n_motion": float(t["MEAN_MOTION"]),
                "bstar":    float(t.get("BSTAR", 0)),
            })
        except (KeyError, ValueError, TypeError):
            continue

    df = pd.DataFrame(records).sort_values("epoch").drop_duplicates("epoch")
    df = df.reset_index(drop=True)
    return df


def quality_filter(df: pd.DataFrame) -> bool:
    """Return True if object passes quality filters."""
    if len(df) < config.MIN_RECORDS_PER_OBJECT:
        return False
    if df["e"].max() > config.MAX_ECCENTRICITY:
        return False
    gaps = df["epoch"].diff().dt.days.dropna()
    if gaps.max() > config.MAX_GAP_DAYS:
        return False
    return True


def interpolate_object(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample TLEs to uniform 24-hour UTC grid using cubic spline interpolation.
    """
    t_orig = (df["epoch"] - df["epoch"].iloc[0]).dt.total_seconds().values
    t_grid = np.arange(0, t_orig[-1] + 1, 86400)  # 24h steps

    result = {"epoch": df["epoch"].iloc[0] + pd.to_timedelta(t_grid, unit="s")}

    for col in ["a_km", "e", "inc", "raan", "aop", "M", "n_motion"]:
        cs = CubicSpline(t_orig, df[col].values)
        result[col] = cs(t_grid)

    # Handle angular wrap-around for RAAN before spline (IMPORTANT)
    # We spline the sin/cos separately then reconstruct
    cs_sin = CubicSpline(t_orig, np.sin(np.radians(df["raan"].values)))
    cs_cos = CubicSpline(t_orig, np.cos(np.radians(df["raan"].values)))
    raan_interp = np.degrees(np.arctan2(cs_sin(t_grid), cs_cos(t_grid))) % 360
    result["raan"] = raan_interp

    return pd.DataFrame(result)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Apply geometric transformation and build 7-dim feature matrix.
    Features: [nx, ny, nz, a_norm, e, n_motion, M_cos, M_sin]
    (We encode M as sin/cos to handle its circular nature too)

    Returns array of shape (T, 7)
    """
    # Geometric transformation: (inc, raan) → (nx, ny, nz)
    n_vec = angular_to_orbit_normal(df["inc"].values, df["raan"].values)

    # Remaining elements (will be normalised by StandardScaler)
    a_km     = df["a_km"].values
    e        = df["e"].values
    n_motion = df["n_motion"].values
    M_cos    = np.cos(np.radians(df["M"].values))
    M_sin    = np.sin(np.radians(df["M"].values))

    features = np.column_stack([
        n_vec[:, 0],   # nx
        n_vec[:, 1],   # ny
        n_vec[:, 2],   # nz
        a_km,          # semi-major axis (km)
        e,             # eccentricity
        n_motion,      # mean motion (rev/day)
        M_cos,         # cos(M) — handles 0°/360° boundary
    ])

    return features


# ═══════════════════════════════════════════════════════════════════
# 4.  Full pipeline: raw JSON → windowed sequences
# ═══════════════════════════════════════════════════════════════════

def load_and_process_all(raw_dir: str = "data/raw",
                          cache_path: str = "data/processed.pkl"):
    """
    Load all per-object JSON files, preprocess, and return
    train/val/test (X, y) arrays.

    X shape: (N_windows, LOOKBACK, 7)
    y shape: (N_windows, 3)   — next-step [nx, ny, nz]
    """
    if os.path.exists(cache_path):
        print(f"Loading cached processed data from '{cache_path}'")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    manifest_path = os.path.join(raw_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No manifest at '{manifest_path}'. Run data/downloader.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    norad_ids = manifest["norad_ids"]
    print(f"Processing {len(norad_ids)} objects...")

    W = config.LOOKBACK_WINDOW

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    # For inverse-transforming predictions to km error
    raw_dfs = []

    n_ok = 0
    for nid in norad_ids:
        fpath = os.path.join(raw_dir, f"{nid}.json")
        if not os.path.exists(fpath):
            continue

        with open(fpath) as f:
            tle_list = json.load(f)

        df = parse_object_tles(tle_list)

        if not quality_filter(df):
            continue

        df_interp = interpolate_object(df)
        features  = build_features(df_interp)          # (T, 7)

        # Build sliding windows
        for t in range(W, len(df_interp)):
            x_window = features[t - W : t]             # (W, 7)
            y_next   = features[t, :3]                 # [nx, ny, nz]

            epoch = df_interp["epoch"].iloc[t]

            if epoch <= pd.Timestamp(config.TRAIN_END):
                X_train.append(x_window)
                y_train.append(y_next)
            elif epoch <= pd.Timestamp(config.VAL_END):
                X_val.append(x_window)
                y_val.append(y_next)
            else:
                X_test.append(x_window)
                y_test.append(y_next)

        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  Processed {n_ok}/{len(norad_ids)} objects")

    print(f"\nObjects used: {n_ok}")

    def to_arr(lst): return np.array(lst, dtype=np.float32)

    X_tr = to_arr(X_train); y_tr = to_arr(y_train)
    X_v  = to_arr(X_val);   y_v  = to_arr(y_val)
    X_te = to_arr(X_test);  y_te = to_arr(y_test)

    # Normalise using TRAINING SET statistics only (prevent leakage)
    # The nx, ny, nz channels are already in [-1,1] but we normalise
    # all channels uniformly for the LSTM
    n_train, W_, F = X_tr.shape
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, F)
    scaler.fit(X_tr_2d)

    X_tr = scaler.transform(X_tr.reshape(-1, F)).reshape(n_train, W_, F)
    X_v  = scaler.transform(X_v.reshape(-1, F)).reshape(-1, W_, F)
    X_te = scaler.transform(X_te.reshape(-1, F)).reshape(-1, W_, F)

    # Save scaler
    os.makedirs("outputs", exist_ok=True)
    with open(config.SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    result = {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_v,  "y_val":   y_v,
        "X_test":  X_te, "y_test":  y_te,
        "scaler":  scaler,
        "n_objects": n_ok,
        "n_train_windows": len(X_tr),
        "n_val_windows":   len(X_v),
        "n_test_windows":  len(X_te),
    }

    # Cache
    Path("data").mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Train windows: {len(X_tr):,}")
    print(f"Val windows:   {len(X_v):,}")
    print(f"Test windows:  {len(X_te):,}")
    print(f"Saved to '{cache_path}'")

    return result


if __name__ == "__main__":
    data = load_and_process_all()
    print("\nFeature matrix stats (training set):")
    print(f"  X_train shape: {data['X_train'].shape}")
    print(f"  y_train shape: {data['y_train'].shape}")

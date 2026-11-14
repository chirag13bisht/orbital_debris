# config.py — Central configuration
# ⚠️  Change your Space-Track password after first use

SPACETRACK_USER     = "bishtchirag13@gmail.com"
SPACETRACK_PASSWORD = "Chirag7838651655"
SPACETRACK_BASE_URL = "https://www.space-track.org"

# ── Dataset settings ─────────────────────────────────────────────
# Cosmos-2251 collision debris (NORAD IDs in range 33594-36000 approx)
# We use the SATCAT object type filtering instead
COSMOS_DEBRIS_EPOCH_START = "2009-02-10"   # 2 weeks after collision
COSMOS_DEBRIS_EPOCH_END   = "2023-12-31"
MIN_RECORDS_PER_OBJECT    = 100            # drop objects with too few TLEs
MAX_GAP_DAYS              = 14             # drop TLEs with gaps > 14 days
MAX_ECCENTRICITY          = 0.1            # LEO only

# ── Data split (chronological — no leakage) ──────────────────────
TRAIN_END   = "2019-05-31"
VAL_END     = "2021-12-31"
# Test: 2022-01-01 to 2023-12-31

# ── Preprocessing ─────────────────────────────────────────────────
INTERP_STEP_HOURS = 24          # resample to uniform 24h grid
LOOKBACK_WINDOW   = 10          # days of history fed to LSTM

# ── Model architecture ────────────────────────────────────────────
LSTM_UNITS_L1     = 128
LSTM_UNITS_L2     = 64
DENSE_UNITS       = 32
DROPOUT_KERNEL    = 0.2
DROPOUT_RECURRENT = 0.1
L2_LAMBDA         = 1e-4
INPUT_DIM         = 7           # [nx, ny, nz, a, e, n_motion, M]
OUTPUT_DIM        = 3           # predict [nx, ny, nz] at t+1

# ── Training ─────────────────────────────────────────────────────
BATCH_SIZE        = 32
MAX_EPOCHS        = 150
EARLY_STOP_PAT    = 15
LR_INITIAL        = 0.001
LR_REDUCE_FACTOR  = 0.5
LR_REDUCE_PAT     = 5
LR_MIN            = 1e-6
RANDOM_SEED       = 42

# ── Evaluation ────────────────────────────────────────────────────
PREDICTION_HORIZON_DAYS = 30
STM_THRESHOLD_KM        = 500   # Space Traffic Management threshold

# ── Output paths ─────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"
MODEL_SAVE_PATH  = "outputs/vector_lstm_model.keras"
SCALER_SAVE_PATH = "outputs/scaler.pkl"
RESULTS_CSV      = "outputs/results.csv"
FIGURES_DIR      = "outputs/figures"

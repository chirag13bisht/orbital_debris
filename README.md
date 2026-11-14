# Space Debris Vector-LSTM — Full Experiment Pipeline

## Project Structure

```
space_debris/
├── config.py                  ← All settings (credentials, hyperparams)
├── main.py                    ← Run everything with one command
├── requirements.txt
├── data/
│   ├── downloader.py          ← Downloads TLEs from Space-Track.org
│   └── preprocessor.py        ← Spline interp + geometric transform
├── model/
│   └── vector_lstm.py         ← Proposed model + Angular-LSTM baseline
├── evaluation/
│   └── metrics.py             ← MAE, RMSE, Wilcoxon, Cohen's d
├── visualization/
│   └── figures.py             ← All 6 publication figures
└── outputs/                   ← Created automatically
    ├── vector_lstm_model.keras
    ├── results.csv
    └── figures/               ← PDF + PNG of all 6 figures
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Register on Space-Track.org (free)

Go to https://www.space-track.org and create a free account.
Then open `config.py` and set:

```python
SPACETRACK_USER     = "your_email@gmail.com"
SPACETRACK_PASSWORD = "your_password"
```

**⚠️ Never share your credentials publicly.**

---

## Step 3 — Run the full pipeline

```bash
python main.py
```

This will automatically:
1. ✅ Login to Space-Track.org and download Cosmos-2251 TLE data
2. ✅ Parse, filter, and apply cubic spline interpolation (24h grid)
3. ✅ Apply geometric transformation (i, Ω) → (nx, ny, nz)
4. ✅ Train the proposed Vector-LSTM model
5. ✅ Train the Angular-LSTM baseline
6. ✅ Run the 6-variant ablation study
7. ✅ Compute MAE, RMSE, σ, P90, Acc@500 on test set
8. ✅ Run Wilcoxon signed-rank test + Cohen's d effect sizes
9. ✅ Run 30-day prediction horizon analysis
10. ✅ Generate all 6 publication-quality PDF figures

---

## Expected outputs

| File | Description |
|------|-------------|
| `outputs/results.csv` | Summary table of all metrics |
| `outputs/all_results.pkl` | Full results including raw errors |
| `outputs/vector_lstm_model.keras` | Trained model |
| `outputs/figures/fig1_training_curves.pdf` | Training convergence |
| `outputs/figures/fig2_wraparound.pdf` | Angular vs vector |
| `outputs/figures/fig3_comparison.pdf` | Baseline comparison |
| `outputs/figures/fig4_ablation.pdf` | Ablation study |
| `outputs/figures/fig5_horizon.pdf` | 30-day horizon analysis |
| `outputs/figures/fig6_distribution.pdf` | Error distribution |

---

## Hardware recommendation

| GPU | Expected training time |
|-----|------------------------|
| NVIDIA RTX 3060 or better | ~2–3 hours |
| Google Colab (free T4 GPU) | ~4–5 hours |
| CPU only | ~8–12 hours |

---

## Running on Google Colab (free GPU)

```python
# In a Colab notebook cell:
!git clone <your_repo_or_upload_files>
%cd space_debris
!pip install -r requirements.txt
!python main.py
```

---

## Troubleshooting

**Space-Track login fails** → Check credentials in config.py.
Your account must be approved (usually instant).

**Out of memory** → Reduce BATCH_SIZE to 16 in config.py.

**Too slow** → Reduce MAX_EPOCHS to 50 for a quick test run.

**No data downloaded** → Space-Track sometimes has maintenance.
Try again after 30 minutes.

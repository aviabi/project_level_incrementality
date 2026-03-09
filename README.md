# Innovation Impact Measurement Pipeline

**End-to-end causal inference system for measuring incremental sales lift using multivariate Bayesian structural time series (mbsts)**

## Overview

This pipeline measures the incremental monthly sales volume contributed by innovation SKUs using observational data, with no randomized experiment required.

### Architecture

- **Python**: Data preparation, orchestration, validation, output storage, visualization
- **R (via rpy2)**: mbsts model fitting and forecasting only
- **Reconciliation**: MiNT (Minimum Trace) with Ledoit-Wolf shrinkage

### Key Features

✅ **Backtesting**: Pre-launch walk-forward validation to assess counterfactual model quality  
✅ **Post-Launch Inference**: Rolling 3-year training windows with monthly updates  
✅ **MiNT Reconciliation**: Coherent SKU-level attribution from global lift  
✅ **Robust Validation**: R² thresholds, MAPE checks, CI coverage assessment  
✅ **7 Diagnostic Plots**: Comprehensive visual diagnostics for model validation  

---

## Installation

### 1. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. R Environment

Install the standalone **mbsts** package in R:

```r
install.packages("mbsts")
install.packages("lubridate")  # Optional, for date handling
```

⚠️ **CRITICAL**: Use the standalone `mbsts` CRAN package, **NOT** `bsts::mbsts`. The APIs are incompatible.

---

## File Structure

```
project_level_incrementality/
│
├── main.py              # Entry point - orchestrates entire pipeline
├── data_prep.py         # Data validation, normalization, fold generation
├── mbsts_model.R        # Pure R mbsts modeling (called via rpy2)
├── inference.py         # De-normalization, delta computation, MiNT reconciliation
├── visualise.py         # All 7 diagnostic plots
├── requirements.txt     # Python package dependencies
└── README.md           # This file
```

---

## Usage

### Quick Start (Demo Mode)

The pipeline includes synthetic data for demonstration:

```bash
python main.py
```

This will:
1. Generate synthetic data (84 months, 2019-01 to 2025-12)
2. Run pre-launch backtest validation
3. Execute post-launch inference (Sep 2023 - Dec 2025)
4. Generate all diagnostic plots
5. Save results to CSV

### Using Your Own Data

Replace the `generate_synthetic_data()` function in `main.py` with your actual data loading:

```python
def load_actual_data():
    """Load your actual df_national DataFrame."""
    df = pd.read_csv("your_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

# In main():
# df_national = generate_synthetic_data()  # Remove this
df_national = load_actual_data()           # Add this
```

### Required Data Format

Your `df_national` DataFrame must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | First day of month (MS frequency) |
| `proxy_A` | float | Median of proxy group A (hl) |
| `proxy_B` | float | Median of proxy group B (hl) |
| `total_portfolio` | float | Actual total sales including innovations (hl) |
| `control_market` | float | Unexposed control market sales (hl) |

**Important**: 
- Use **median** aggregation for proxy groups (robust to outliers)
- No date gaps allowed (monthly continuity required)
- All series must have non-zero variance

### Configuration

Edit the `config` dictionary in `main.py`:

```python
config = {
    "sku1_launch": "2023-03-01",           # Launch date
    "sku2_launch": "2023-03-01",           # Launch date (second SKU)
    "stabilisation_months": 6,             # Post-launch stabilization period
    "first_analysis_month": "2023-09-01",  # First analysis month
    "latest_analysis_month": "2025-12-01", # Last analysis month
    "training_years": 3,                   # Rolling training window size
    "proxy_cols": ["proxy_A", "proxy_B"],  # Proxy column names
    "inno_names": ["BrutalFruit_Can500", "BrutalFruit_NRB275"],  # SKU names
    "control_col": "control_market",       # Control regressor column
    "actual_col": "total_portfolio",       # Actual sales column
    "n_seasons": 12,                       # Seasonal periods (monthly)
    "niter": 1000,                         # MCMC iterations
    "burnin": 200,                         # Burn-in period
    "ci_level": 0.80,                      # Credible interval level
    "backtest_min_train": 12,              # Minimum backtest training months
    "backtest_horizon": 3,                 # Backtest forecast steps
    "backtest_step": 1                     # Backtest fold step size
}
```

---

## Outputs

### CSV Files

1. **`results_innovation_impact.csv`** - Post-launch monthly results
   - Columns: month, delta_mean, delta_lower, delta_upper, prob_positive
   - SKU-level attribution: delta_SKU1, delta_SKU2, share_SKU1, share_SKU2
   - Model diagnostics: r2_proxy_A, r2_proxy_B, coherence_err

2. **`backtest_results.csv`** - Pre-launch validation results
   - One row per fold per horizon step per proxy group
   - Columns: fold_id, train_end, forecast_month, actual, pred_mean, pred_lower, pred_upper
   - Metrics: mae, mape, rmse, coverage, r2_A, r2_B

### Plots

#### Post-Launch Analysis

1. **`plot_delta_global.png`** - Incremental lift over time with 80% CI
2. **`plot_sku_attribution.png`** - Stacked bar chart of SKU-level attribution
3. **`plot_model_diagnostics.png`** - In-sample R² trends by proxy group

#### Backtest Validation

4. **`plot_backtest_fit.png`** - Actual vs predicted for each proxy group by horizon
5. **`plot_mape_by_horizon.png`** - Mean absolute percentage error by forecast step
6. **`plot_ci_coverage.png`** - Credible interval coverage rates
7. **`plot_r2_over_folds.png`** - Out-of-sample R² stability across folds

---

## Methodology

### Backtesting Strategy

**Purpose**: Validate the counterfactual model *before* the innovation launch using only pre-launch data.

**Approach**: Expanding-window walk-forward cross-validation
- Minimum initial training: 12 months (1 seasonal cycle)
- Forecast horizon: 3 months ahead
- Step size: 1 month (folds overlap)
- Stop condition: Any forecast month ≥ launch date

**Evaluation Metrics**:
- MAPE, MAE, RMSE (per horizon step, per proxy group)
- 80% CI coverage rate
- Out-of-sample R² (R²_pred, can be negative)

**Acceptance Thresholds**:
- ⚠️ WARNING: MAPE > 15%, Coverage < 70%, Median R² < 0.70
- 🚨 CRITICAL: Median R² < 0.00 (worse than naive mean)

### Post-Launch Inference

**Training Window**: Rolling 3-year window (36 months)
- Normalizes using training statistics only (no data leakage)
- Trims rows to avoid KFAS seasonal boundary errors

**mbsts Model Components**:
- Local linear trend
- Seasonal (12 periods)
- Regression on control market (or fallback if variance too low)

**Counterfactual Logic**:
```
Δ_global = Actual - (Proxy_A_cf + Proxy_B_cf)
```

**MiNT Reconciliation**:
```
W = LedoitWolf(residuals)
w = inv(W) @ 1 / (1' @ inv(W) @ 1)
Δ_SKU = w * Δ_global
```

Ensures coherence: `sum(Δ_SKU) = Δ_global` (within 0.01 tolerance)

---

## Validation Rules

### Skip Conditions

A month/fold is skipped if:
- Training rows < 18 (after trimming)
- Proxy variance = 0 in training window
- Analysis month data missing

### Warnings

- R² < 0.70: Insufficient model accuracy
- R² < 0.85: Acceptable but not ideal (printed as notice)
- MAPE > 15%: Prediction error too high
- Coverage < 70%: Credible intervals poorly calibrated
- Coherence error ≥ 0.01: MiNT reconciliation failed (raises ValueError)

### Critical Errors

- R² < 0.00: Model worse than naive mean (RuntimeWarning, continues execution)
- Control market variance = 0: Falls back to `(proxy_A + proxy_B) / 2`

---

## Troubleshooting

### R Package Issues

**Error**: `Failed to load mbsts package`

**Solution**:
```r
# In R console
install.packages("mbsts")
library(mbsts)  # Verify it loads
```

### rpy2 Connection Issues

**Error**: `R not found` or `R_HOME not set`

**Solution** (Windows):
```powershell
# Set R_HOME environment variable
$env:R_HOME = "C:\Program Files\R\R-4.3.0"  # Adjust to your R installation
```

**Solution** (Linux/Mac):
```bash
export R_HOME=/usr/lib/R
```

### KFAS Seasonal Errors

**Error**: `Error in KFAS seasonal component`

**Cause**: Training rows not properly trimmed for seasonal initialization

**Solution**: The pipeline automatically trims rows using:
```python
n_trimmed = floor((n + 1) / n_seasons) * n_seasons - 1
```

If errors persist, increase `backtest_min_train` or `training_years`.

### Memory Issues (Large Datasets)

**Solution**: Reduce MCMC iterations:
```python
config["niter"] = 500   # Down from 1000
config["burnin"] = 100  # Down from 200
```

---

## Technical Notes

### mbsts vs bsts

**DO NOT CONFUSE** these two packages:

| Package | Namespace | Function | Status |
|---------|-----------|----------|--------|
| `mbsts` | Standalone | `mbsts_function()`, `tsc.setting()` | ✅ **USED** |
| `bsts` | `bsts::` | `bsts()`, `AddLocalLinearTrend()` | ❌ **NOT USED** |

The `bsts` package has a different API and is **incompatible** with this pipeline.

### Posterior Shape Convention

- **Single-step**: `(draws, 2)` — rows are MCMC draws, columns are proxy groups
- **Multi-step**: `(draws, 2, steps)` — 3D array with time in 3rd dimension

### Normalization Strategy

All series are Z-score normalized using **training window statistics only**:
```
x_norm = (x - μ_train) / σ_train
```

This prevents data leakage and stabilizes numerical optimization in mbsts.

---

## References

- **mbsts R Package**: [CRAN - mbsts](https://cran.r-project.org/package=mbsts)
- **MiNT Reconciliation**: Wickramasuriya et al. (2019), "Optimal forecast reconciliation for hierarchical and grouped time series"
- **Ledoit-Wolf Shrinkage**: Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional covariance matrices"
- **Bayesian Structural Time Series**: Scott & Varian (2014), "Predicting the Present with Bayesian Structural Time Series"

---

## Support

For issues related to:
- **mbsts package**: Check [mbsts documentation](https://cran.r-project.org/package=mbsts)
- **rpy2 integration**: See [rpy2 docs](https://rpy2.github.io/)
- **Pipeline logic**: Review comments in source code files

---

## License

MIT License - feel free to adapt for your use case.

---

**Built with**: Python 3.10+, R 4.2+, mbsts, rpy2, scikit-learn, matplotlib

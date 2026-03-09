"""
main.py

Entry point for end-to-end innovation impact measurement pipeline.
Orchestrates: validation, backtesting, post-launch inference, visualization.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

# Local modules
from data_prep import (
    validate_and_prepare,
    generate_backtest_folds,
    prepare_monthly_data,
    prepare_backtest_fold_data
)
from inference import (
    denormalize_posterior,
    denormalize_fitted,
    compute_delta_global,
    compute_ledoit_wolf_weights,
    mint_reconcile,
    evaluate_backtest_fold,
    compute_oos_r2,
    validate_r2
)
from visualise import generate_all_plots

# Activate rpy2 converters
numpy2ri.activate()
pandas2ri.activate()


def setup_r_environment():
    """
    Initialize R environment and source mbsts_model.R.
    
    Returns:
        R function handle for run_mbsts_month
    """
    print("\n" + "═"*60)
    print("INITIALIZING R ENVIRONMENT")
    print("═"*60)
    
    # Import required R packages
    try:
        ro.r('library(mbsts)')
        print("✓ Loaded R package: mbsts")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load mbsts package. Please install it in R:\n"
            f"  install.packages('mbsts')\n"
            f"Error: {str(e)}"
        )
    
    # Source the R model file
    r_script_path = Path(__file__).parent / "mbsts_model.R"
    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found: {r_script_path}")
    
    ro.r(f'source("{str(r_script_path).replace(chr(92), "/")}")')
    print(f"✓ Sourced: {r_script_path.name}")
    
    # Get R function handle
    run_mbsts_month = ro.globalenv['run_mbsts_month']
    
    return run_mbsts_month


def run_backtest(df_national, config, run_mbsts_month):
    """
    Execute walk-forward backtest on pre-launch data.
    
    Args:
        df_national: Validated national DataFrame
        config: Configuration dictionary
        run_mbsts_month: R function handle
        
    Returns:
        Tuple of (backtest_results_df, backtest_summary_df)
    """
    print("\n" + "═"*60)
    print("BACKTESTING — PRE-LAUNCH VALIDATION")
    print("═"*60)
    
    # Generate folds
    folds = generate_backtest_folds(df_national, config)
    
    if len(folds) == 0:
        warnings.warn("No valid backtest folds generated. Skipping backtest.", RuntimeWarning)
        return pd.DataFrame(), pd.DataFrame()
    
    all_results = []
    
    for fold in folds:
        fold_id = fold["fold_id"]
        print(f"\n--- Fold {fold_id}/{len(folds)} ---")
        print(f"  Train: {fold['train_start'].date()} to {fold['train_end'].date()} ({fold['n_train']} months)")
        print(f"  Forecast: {fold['forecast_dates'][0].date()} to {fold['forecast_dates'][-1].date()}")
        
        try:
            # Prepare data
            Y_train, X_train, X_test, norm_params, n_trimmed = prepare_backtest_fold_data(
                df_national, fold, config["proxy_cols"], config["control_col"], config["n_seasons"]
            )
            
            print(f"  Training rows (after trim): {n_trimmed}")
            
            # Call R model
            r_result = run_mbsts_month(
                Y_train, X_train, X_test,
                mc=config["niter"],
                burn=config["burnin"],
                n_seasons=config["n_seasons"],
                steps=config["backtest_horizon"]
            )
            
            # Extract results
            pred_dist = np.array(r_result.rx2('pred_dist'))
            fitted_vals = np.array(r_result.rx2('fitted_vals'))
            group_r2 = np.array(r_result.rx2('group_r2'))
            
            print(f"  In-sample R²: {group_r2}")
            
            # De-normalize
            pred_dist_denorm = denormalize_posterior(pred_dist, norm_params, config["proxy_cols"])
            fitted_vals_denorm = denormalize_fitted(fitted_vals, norm_params, config["proxy_cols"])
            
            # Get actual values for forecast period
            actual_rows = df_national[df_national["date"].isin(fold["forecast_dates"])].copy()
            actual_rows = actual_rows.sort_values("date")
            actual_vals = actual_rows[config["proxy_cols"]].values  # (steps, 2)
            
            # Evaluate fold
            fold_metrics = evaluate_backtest_fold(
                pred_dist_denorm, actual_vals, norm_params, config["proxy_cols"], config["ci_level"]
            )
            
            # Compute out-of-sample R²
            pred_means = np.array([[m["pred_mean"] for m in fold_metrics if m["group"] == col] 
                                   for col in config["proxy_cols"]]).T
            r2_A, r2_B = compute_oos_r2(actual_vals, pred_means)
            
            # Store results
            for i, h_date in enumerate(fold["forecast_dates"]):
                for g, col in enumerate(config["proxy_cols"]):
                    metric_idx = i * len(config["proxy_cols"]) + g
                    m = fold_metrics[metric_idx]
                    
                    all_results.append({
                        "fold_id": fold_id,
                        "train_end": fold["train_end"],
                        "n_train": n_trimmed,
                        "horizon_step": i + 1,
                        "forecast_month": h_date,
                        "group": col,
                        "actual": m["actual"],
                        "pred_mean": m["pred_mean"],
                        "pred_lower": m["pred_lower"],
                        "pred_upper": m["pred_upper"],
                        "mae": m["mae"],
                        "mape": m["mape"],
                        "rmse": m["rmse"],
                        "coverage": m["coverage"],
                        "r2_A": r2_A,
                        "r2_B": r2_B
                    })
            
            print(f"  Out-of-sample R²: A={r2_A:.4f}, B={r2_B:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Fold {fold_id} failed: {str(e)}")
            warnings.warn(f"Fold {fold_id} failed: {str(e)}", RuntimeWarning)
            continue
    
    if len(all_results) == 0:
        warnings.warn("All backtest folds failed. No backtest results.", RuntimeWarning)
        return pd.DataFrame(), pd.DataFrame()
    
    # Create results DataFrame
    backtest_df = pd.DataFrame(all_results)
    
    # Compute summary statistics
    summary = compute_backtest_summary(backtest_df, config)
    
    # Save results
    backtest_df.to_csv("backtest_results.csv", index=False)
    print(f"\n✓ Saved: backtest_results.csv ({len(backtest_df)} rows)")
    
    return backtest_df, summary


def compute_backtest_summary(backtest_df, config):
    """
    Compute aggregate backtest metrics.
    
    Args:
        backtest_df: Backtest results DataFrame
        config: Configuration dictionary
        
    Returns:
        Summary DataFrame grouped by horizon_step
    """
    print("\n" + "═"*60)
    print("BACKTEST SUMMARY")
    print("═"*60)
    
    # Group by horizon step
    summary_rows = []
    
    for h in sorted(backtest_df["horizon_step"].unique()):
        h_data = backtest_df[backtest_df["horizon_step"] == h]
        
        for col_idx, col in enumerate(config["proxy_cols"]):
            col_data = h_data[h_data["group"] == col]
            
            if col_idx == 0:
                row = {"horizon_step": h}
            
            suffix = "_A" if col_idx == 0 else "_B"
            row[f"mean_mae{suffix}"] = col_data["mae"].mean()
            row[f"mean_mape{suffix}"] = col_data["mape"].mean()
            row[f"mean_rmse{suffix}"] = col_data["rmse"].mean()
            row[f"mean_coverage{suffix}"] = col_data["coverage"].mean()
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Fold-level R² summary
    fold_data = backtest_df.drop_duplicates(subset=["fold_id"])
    
    median_r2_A = fold_data["r2_A"].median()
    median_r2_B = fold_data["r2_B"].median()
    
    pct_above_070_A = (fold_data["r2_A"] > 0.70).mean() * 100
    pct_above_070_B = (fold_data["r2_B"] > 0.70).mean() * 100
    
    # Print summary
    print("\n--- Metrics by Horizon Step ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- Fold-Level R² Summary ---")
    print(f"  Median R² (Proxy A): {median_r2_A:.4f}")
    print(f"  Median R² (Proxy B): {median_r2_B:.4f}")
    print(f"  % Folds R²>0.70 (A): {pct_above_070_A:.1f}%")
    print(f"  % Folds R²>0.70 (B): {pct_above_070_B:.1f}%")
    
    # Validation checks
    print("\n--- Validation Checks ---")
    
    for suffix in ["_A", "_B"]:
        if summary_df[f"mean_mape{suffix}"].max() > 15:
            print(f"  ⚠ WARNING: Max MAPE{suffix} > 15% (insufficient accuracy)")
        
        if summary_df[f"mean_coverage{suffix}"].min() < 0.70:
            print(f"  ⚠ WARNING: Min coverage{suffix} < 70% (poor CI calibration)")
    
    if median_r2_A < 0.70 or median_r2_B < 0.70:
        print(f"  ⚠ WARNING: Median R² < 0.70 (insufficient predictive accuracy)")
    
    if median_r2_A < 0.00 or median_r2_B < 0.00:
        print(f"  ⚠ CRITICAL: Median R² < 0.00 (worse than naive mean)")
        warnings.warn("Backtest median R² < 0.00 - model worse than naive", RuntimeWarning)
    
    if pct_above_070_A < 60 or pct_above_070_B < 60:
        print(f"  ⚠ WARNING: <60% of folds with R²>0.70 (model unstable)")
    
    return summary_df


def run_analysis_loop(df_national, config, run_mbsts_month):
    """
    Execute post-launch inference for all analysis months.
    
    Args:
        df_national: Validated national DataFrame
        config: Configuration dictionary
        run_mbsts_month: R function handle
        
    Returns:
        Results DataFrame with one row per analysis month
    """
    print("\n" + "═"*60)
    print("POST-LAUNCH INFERENCE")
    print("═"*60)
    
    analysis_months = pd.date_range(
        start=config["first_analysis_month"],
        end=config["latest_analysis_month"],
        freq="MS"
    )
    
    print(f"Analyzing {len(analysis_months)} months from {analysis_months[0].date()} to {analysis_months[-1].date()}")
    
    results = []
    
    for analysis_month in analysis_months:
        print(f"\n--- {analysis_month.strftime('%b %Y')} ---")
        
        try:
            # Prepare data
            Y_train, X_train, X_test, norm_params, n_trimmed = prepare_monthly_data(
                df_national, analysis_month, config["training_years"],
                config["proxy_cols"], config["control_col"], config["n_seasons"]
            )
            
            # Extract actual total portfolio for this month
            actual = df_national.loc[
                df_national["date"] == analysis_month, config["actual_col"]
            ].values[0]
            
            print(f"  Training: {n_trimmed} months")
            print(f"  Actual total portfolio: {actual:.2f} hl")
            
            # Call R model
            r_result = run_mbsts_month(
                Y_train, X_train, X_test,
                mc=config["niter"],
                burn=config["burnin"],
                n_seasons=config["n_seasons"],
                steps=1
            )
            
            # Extract results
            pred_dist = np.array(r_result.rx2('pred_dist'))  # (draws, 2)
            fitted_vals = np.array(r_result.rx2('fitted_vals'))  # (n_train, 2)
            group_r2 = np.array(r_result.rx2('group_r2'))  # (2,)
            
            # Validate R²
            all_r2_ok = validate_r2(group_r2, analysis_month.strftime('%b %Y'))
            
            # De-normalize
            v_cf_draws = denormalize_posterior(pred_dist, norm_params, config["proxy_cols"])
            fitted_vals_denorm = denormalize_fitted(fitted_vals, norm_params, config["proxy_cols"])
            
            # Compute delta global
            delta_mean, delta_lower, delta_upper, prob_positive = compute_delta_global(
                v_cf_draws, actual
            )
            
            print(f"  Δ_global: {delta_mean:.2f} hl [80% CI: {delta_lower:.2f}, {delta_upper:.2f}]")
            print(f"  P(Δ>0): {prob_positive:.3f}")
            
            # De-normalize Y_train for residual calculation
            train_end = analysis_month - pd.DateOffset(months=1)
            train_start = train_end - pd.DateOffset(years=config["training_years"])
            train_mask = (df_national["date"] >= train_start) & (df_national["date"] <= train_end)
            train_df = df_national[train_mask].reset_index(drop=True)
            
            # Trim to match fitted_vals
            n_train_raw = len(train_df)
            n_seasons = config["n_seasons"]
            n_trimmed_check = int(np.floor((n_train_raw + 1) / n_seasons) * n_seasons - 1)
            rows_to_drop = n_train_raw - n_trimmed_check
            train_df_trimmed = train_df.iloc[rows_to_drop:].reset_index(drop=True)
            
            y_train_actual = train_df_trimmed[config["proxy_cols"]].values
            
            # MiNT reconciliation
            w, W = compute_ledoit_wolf_weights(fitted_vals_denorm, y_train_actual)
            delta_inno, shares, coherence_err = mint_reconcile(
                delta_mean, w, config["inno_names"]
            )
            
            print(f"  SKU attribution:")
            for i, sku in enumerate(config["inno_names"]):
                print(f"    {sku}: {delta_inno[i]:.2f} hl ({shares[i]*100:.1f}%)")
            print(f"  Coherence error: {coherence_err:.6f}")
            
            # Store results
            results.append({
                "month": analysis_month,
                "month_label": analysis_month.strftime("%b %Y"),
                "train_start": train_start,
                "train_end": train_end,
                "n_train_months": n_trimmed,
                "delta_mean": delta_mean,
                "delta_lower": delta_lower,
                "delta_upper": delta_upper,
                "prob_positive": prob_positive,
                "vcf_mean": v_cf_draws.sum(axis=1).mean(),
                "actual_mean": actual,
                "delta_inno": delta_inno,
                "shares": shares,
                "coherence_err": coherence_err,
                "group_r2": group_r2,
                "all_r2_ok": all_r2_ok
            })
            
        except Exception as e:
            print(f"  ⚠ Analysis failed: {str(e)}")
            warnings.warn(f"Month {analysis_month.date()} failed: {str(e)}", RuntimeWarning)
            continue
    
    if len(results) == 0:
        raise RuntimeError("All analysis months failed. No results to save.")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Prepare output DataFrame (flatten arrays for CSV)
    output_df = results_df.copy()
    
    # Flatten delta_inno and shares
    for i, sku in enumerate(config["inno_names"]):
        output_df[f"delta_{sku}"] = output_df["delta_inno"].apply(lambda x: x[i])
        output_df[f"share_{sku}"] = output_df["shares"].apply(lambda x: x[i])
    
    # Flatten group_r2
    for i, col in enumerate(config["proxy_cols"]):
        output_df[f"r2_{col}"] = output_df["group_r2"].apply(lambda x: x[i])
    
    # Drop array columns
    output_df = output_df.drop(columns=["delta_inno", "shares", "group_r2"])
    
    # Save
    output_df.to_csv("results_innovation_impact.csv", index=False)
    print(f"\n✓ Saved: results_innovation_impact.csv ({len(output_df)} rows)")
    
    return results_df


def main():
    """
    Main entry point for the pipeline.
    """
    print("\n" + "═"*60)
    print("INNOVATION IMPACT MEASUREMENT PIPELINE")
    print("South Africa — mbsts Causal Inference")
    print("═"*60)
    
    # Configuration
    config = {
        "sku1_launch": "2023-03-01",
        "sku2_launch": "2023-03-01",
        "stabilisation_months": 6,
        "first_analysis_month": "2023-09-01",
        "latest_analysis_month": "2025-12-01",
        "training_years": 3,
        "proxy_cols": ["proxy_A", "proxy_B"],
        "inno_names": ["BrutalFruit_Can500", "BrutalFruit_NRB275"],
        "control_col": "control_market",
        "actual_col": "total_portfolio",
        "n_seasons": 12,
        "niter": 1000,
        "burnin": 200,
        "ci_level": 0.80,
        "w_shrinkage": "ledoit_wolf",
        "backtest_min_train": 12,
        "backtest_horizon": 3,
        "backtest_step": 1
    }
    
    # Example data (replace with actual data load)
    print("\n⚠ DEMO MODE: Using synthetic data")
    print("   Replace this with your actual df_national DataFrame")
    
    df_national = generate_synthetic_data()
    
    # Step 1: Validate and prepare
    df_clean = validate_and_prepare(df_national)
    
    # Step 2: Setup R environment
    run_mbsts_month = setup_r_environment()
    
    # Step 3: Run backtest
    backtest_df, backtest_summary = run_backtest(df_clean, config, run_mbsts_month)
    
    # Step 4: Run post-launch analysis
    results_df = run_analysis_loop(df_clean, config, run_mbsts_month)
    
    # Step 5: Generate visualizations
    if len(backtest_df) > 0 and len(results_df) > 0:
        generate_all_plots(results_df, backtest_df, backtest_summary, config)
    else:
        print("\n⚠ Skipping visualization due to insufficient results")
    
    print("\n" + "═"*60)
    print("PIPELINE COMPLETE")
    print("═"*60)
    print("\nOutput files:")
    print("  - results_innovation_impact.csv")
    print("  - backtest_results.csv")
    print("  - plot_delta_global.png")
    print("  - plot_sku_attribution.png")
    print("  - plot_model_diagnostics.png")
    print("  - plot_backtest_fit.png")
    print("  - plot_mape_by_horizon.png")
    print("  - plot_ci_coverage.png")
    print("  - plot_r2_over_folds.png")


def generate_synthetic_data():
    """
    Generate synthetic data for demonstration purposes.
    Replace this with your actual data loading logic.
    
    Returns:
        Synthetic df_national DataFrame
    """
    np.random.seed(42)
    
    dates = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    n = len(dates)
    
    # Synthetic series with trend and seasonality
    t = np.arange(n)
    trend = 1000 + 5 * t
    seasonal = 100 * np.sin(2 * np.pi * t / 12)
    
    proxy_A = trend + seasonal + np.random.normal(0, 50, n)
    proxy_B = trend * 0.8 + seasonal * 0.7 + np.random.normal(0, 40, n)
    control_market = trend * 1.2 + seasonal * 0.5 + np.random.normal(0, 60, n)
    
    # Add innovation lift post-launch
    launch_idx = np.where(dates >= pd.Timestamp("2023-03-01"))[0]
    innovation_lift = np.zeros(n)
    innovation_lift[launch_idx] = np.linspace(0, 200, len(launch_idx))
    
    total_portfolio = proxy_A + proxy_B + innovation_lift
    
    df = pd.DataFrame({
        "date": dates,
        "proxy_A": proxy_A,
        "proxy_B": proxy_B,
        "control_market": control_market,
        "total_portfolio": total_portfolio
    })
    
    return df


if __name__ == "__main__":
    main()

"""
visualise.py

Visualization functions for innovation impact measurement and backtest evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def plot_delta_global(results_df: pd.DataFrame, output_path: str = "plot_delta_global.png"):
    """
    Plot 1: Incremental sales lift over time with 80% CI band.
    
    Args:
        results_df: DataFrame with columns: month, delta_mean, delta_lower, delta_upper
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(results_df["month"])
    
    # Main line
    ax.plot(dates, results_df["delta_mean"], 
            color="#2E86AB", linewidth=2.5, label="Incremental Lift (Δ)", marker='o')
    
    # Shaded CI band
    ax.fill_between(dates, results_df["delta_lower"], results_df["delta_upper"],
                    alpha=0.25, color="#2E86AB", label="80% Credible Interval")
    
    # Zero reference line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel("Analysis Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Incremental Sales (hl)", fontsize=12, fontweight="bold")
    ax.set_title("Incremental Sales Lift — ZAF Innovation Portfolio", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_sku_attribution(results_df: pd.DataFrame, inno_names: List[str], 
                        output_path: str = "plot_sku_attribution.png"):
    """
    Plot 2: Stacked bar chart of SKU-level attribution over time.
    
    Args:
        results_df: DataFrame with month and delta_inno columns
        inno_names: List of innovation SKU names
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(results_df["month"])
    
    # Extract delta_inno arrays (stored as numpy arrays)
    deltas = np.array([row for row in results_df["delta_inno"]])
    
    # Colors for each SKU
    colors = ["#A23B72", "#F18F01"]
    
    # Stacked bar chart
    bar_width = 20  # days
    bottom = np.zeros(len(dates))
    
    for i, sku_name in enumerate(inno_names):
        ax.bar(dates, deltas[:, i], width=bar_width, bottom=bottom,
               label=sku_name, color=colors[i % len(colors)], alpha=0.85)
        bottom += deltas[:, i]
    
    # Formatting
    ax.set_xlabel("Analysis Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Incremental Sales (hl)", fontsize=12, fontweight="bold")
    ax.set_title("MiNT-Reconciled SKU-Level Attribution", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":", axis='y')
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_model_diagnostics(results_df: pd.DataFrame, 
                          output_path: str = "plot_model_diagnostics.png"):
    """
    Plot 3: In-sample R² by proxy group over analysis months.
    
    Args:
        results_df: DataFrame with month and group_r2 columns
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(results_df["month"])
    
    # Extract R² arrays
    r2_values = np.array([row for row in results_df["group_r2"]])
    
    # Plot lines for each proxy group
    ax.plot(dates, r2_values[:, 0], 
            color="#06A77D", linewidth=2, marker='o', label="Proxy A")
    ax.plot(dates, r2_values[:, 1], 
            color="#D4A5A5", linewidth=2, marker='s', label="Proxy B")
    
    # Threshold lines
    ax.axhline(y=0.70, color="orange", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Warning Threshold (R²=0.70)")
    ax.axhline(y=0.85, color="green", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Good Threshold (R²=0.85)")
    
    # Formatting
    ax.set_xlabel("Analysis Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("In-Sample R²", fontsize=12, fontweight="bold")
    ax.set_title("Model Diagnostics: In-Sample R² by Proxy Group (Post-Launch)", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(0, 1.05)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_backtest_fit(backtest_df: pd.DataFrame, proxy_cols: List[str],
                     output_path: str = "plot_backtest_fit.png"):
    """
    Plot 4: Backtest actual vs predicted with 80% CI, color-coded by horizon step.
    
    Args:
        backtest_df: DataFrame with fold results
        proxy_cols: List of proxy column names
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = {1: "#023047", 2: "#219EBC", 3: "#8ECAE6"}  # Darkest to lightest
    
    for idx, proxy in enumerate(proxy_cols):
        ax = axes[idx]
        
        # Filter data for this proxy
        proxy_data = backtest_df[backtest_df["group"] == proxy].copy()
        proxy_data = proxy_data.sort_values("forecast_month")
        
        dates = pd.to_datetime(proxy_data["forecast_month"])
        
        # Plot by horizon step
        for h in [1, 2, 3]:
            h_data = proxy_data[proxy_data["horizon_step"] == h]
            h_dates = pd.to_datetime(h_data["forecast_month"])
            
            ax.plot(h_dates, h_data["actual"], 
                   color="black", linewidth=2, linestyle="-", 
                   marker='o' if h == 1 else None,
                   label="Actual" if h == 1 else None)
            
            ax.plot(h_dates, h_data["pred_mean"], 
                   color=colors[h], linewidth=1.5, linestyle="--", 
                   marker='x', alpha=0.8, label=f"Predicted (h={h})")
            
            ax.fill_between(h_dates, h_data["pred_lower"], h_data["pred_upper"],
                          alpha=0.15, color=colors[h])
        
        # Formatting
        ax.set_ylabel(f"{proxy} (hl)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title(f"Proxy Group: {proxy}", fontsize=12, fontweight="bold")
    
    axes[-1].set_xlabel("Forecast Month", fontsize=12, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    
    fig.suptitle("Walk-Forward Backtest: Counterfactual Fit (Pre-Launch)", 
                fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_mape_by_horizon(summary_df: pd.DataFrame, 
                        output_path: str = "plot_mape_by_horizon.png"):
    """
    Plot 5: MAPE by horizon step, grouped bar chart.
    
    Args:
        summary_df: Summary DataFrame grouped by horizon_step
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = summary_df["horizon_step"].values
    mape_A = summary_df["mean_mape_A"].values
    mape_B = summary_df["mean_mape_B"].values
    
    x = np.arange(len(horizons))
    width = 0.35
    
    ax.bar(x - width/2, mape_A, width, label="Proxy A", color="#06A77D", alpha=0.85)
    ax.bar(x + width/2, mape_B, width, label="Proxy B", color="#D4A5A5", alpha=0.85)
    
    # Threshold line
    ax.axhline(y=15, color="red", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Warning Threshold (15%)")
    
    # Formatting
    ax.set_xlabel("Forecast Horizon (steps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean MAPE (%)", fontsize=12, fontweight="bold")
    ax.set_title("Backtest MAPE by Forecast Horizon", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":", axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_ci_coverage(summary_df: pd.DataFrame, 
                    output_path: str = "plot_ci_coverage.png"):
    """
    Plot 6: CI coverage rate by horizon step.
    
    Args:
        summary_df: Summary DataFrame grouped by horizon_step
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = summary_df["horizon_step"].values
    cov_A = summary_df["mean_coverage_A"].values * 100  # Convert to percentage
    cov_B = summary_df["mean_coverage_B"].values * 100
    
    x = np.arange(len(horizons))
    width = 0.35
    
    ax.bar(x - width/2, cov_A, width, label="Proxy A", color="#06A77D", alpha=0.85)
    ax.bar(x + width/2, cov_B, width, label="Proxy B", color="#D4A5A5", alpha=0.85)
    
    # Threshold line
    ax.axhline(y=70, color="orange", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Warning Threshold (70%)")
    ax.axhline(y=80, color="green", linestyle="--", linewidth=1.5, 
               alpha=0.6, label="Target (80%)")
    
    # Formatting
    ax.set_xlabel("Forecast Horizon (steps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("80% CI Coverage Rate by Horizon", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":", axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_r2_over_folds(backtest_df: pd.DataFrame, 
                      output_path: str = "plot_r2_over_folds.png"):
    """
    Plot 7: Out-of-sample R² stability across backtest folds.
    
    Args:
        backtest_df: DataFrame with fold-level R² values
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique fold data (one row per fold)
    fold_data = backtest_df.drop_duplicates(subset=["fold_id"]).copy()
    fold_data = fold_data.sort_values("train_end")
    
    dates = pd.to_datetime(fold_data["train_end"])
    
    # Plot R² lines
    ax.plot(dates, fold_data["r2_A"], 
            color="#06A77D", linewidth=2, marker='o', label="Proxy A")
    ax.plot(dates, fold_data["r2_B"], 
            color="#D4A5A5", linewidth=2, marker='s', label="Proxy B")
    
    # Threshold lines
    ax.axhline(y=0.70, color="orange", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Warning Threshold (R²=0.70)")
    ax.axhline(y=0.85, color="green", linestyle="--", linewidth=1.5, 
               alpha=0.7, label="Good Threshold (R²=0.85)")
    ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.5, 
               alpha=0.6, label="Naive Baseline (R²=0)")
    
    # Shade warning region
    ax.fill_between(dates, -1, 0.70, alpha=0.08, color="red")
    
    # Formatting
    ax.set_xlabel("Fold Training End Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Out-of-Sample R²", fontsize=12, fontweight="bold")
    ax.set_title("Out-of-Sample R² Stability Across Backtest Folds", 
                 fontsize=14, fontweight="bold", pad=20)
    
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(-0.2, 1.05)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def generate_all_plots(
    results_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    config: Dict
):
    """
    Generate all 7 plots for the analysis.
    
    Args:
        results_df: Post-launch results DataFrame
        backtest_df: Backtest results DataFrame (row per fold per horizon)
        backtest_summary: Backtest summary DataFrame (grouped by horizon)
        config: Configuration dictionary
    """
    print("\n" + "═"*60)
    print("GENERATING VISUALIZATIONS")
    print("═"*60)
    
    try:
        # Post-launch plots
        plot_delta_global(results_df)
        plot_sku_attribution(results_df, config["inno_names"])
        plot_model_diagnostics(results_df)
        
        # Backtest plots
        plot_backtest_fit(backtest_df, config["proxy_cols"])
        plot_mape_by_horizon(backtest_summary)
        plot_ci_coverage(backtest_summary)
        plot_r2_over_folds(backtest_df)
        
        print("\n✓ All 7 plots generated successfully")
        
    except Exception as e:
        print(f"\n⚠ Error generating plots: {str(e)}")
        raise

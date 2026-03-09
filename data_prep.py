"""
data_prep.py

Data validation, normalization, and backtest fold generation for innovation impact measurement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


def validate_and_prepare(df_national: pd.DataFrame) -> pd.DataFrame:
    """
    Validate input data and prepare for analysis.
    
    Args:
        df_national: Raw national-level dataframe
        
    Returns:
        Clean, sorted DataFrame
        
    Raises:
        ValueError: If required columns missing or data validation fails
    """
    required_cols = ["date", "proxy_A", "proxy_B", "total_portfolio", "control_market"]
    
    # 1. Assert required columns exist
    missing = set(required_cols) - set(df_national.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # 2. Convert date to datetime and sort
    df = df_national.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 3. Validate no monthly date gaps
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS"
    )
    
    actual_dates = set(df["date"])
    expected_set = set(expected_dates)
    
    if actual_dates != expected_set:
        missing_dates = expected_set - actual_dates
        extra_dates = actual_dates - expected_set
        
        msg = []
        if missing_dates:
            msg.append(f"Missing dates: {sorted(missing_dates)[:5]}")
        if extra_dates:
            msg.append(f"Extra dates: {sorted(extra_dates)[:5]}")
        
        raise ValueError(f"Date sequence validation failed. {' '.join(msg)}")
    
    # 4. Validate non-zero variance for proxy and control columns
    value_cols = ["proxy_A", "proxy_B", "control_market"]
    for col in value_cols:
        std = df[col].std()
        if std < 1e-6:
            raise ValueError(
                f"Column '{col}' has zero or near-zero variance (std={std:.2e}). "
                f"Cannot use for modeling."
            )
    
    print(f"✓ Validated {len(df)} monthly observations from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"✓ All required columns present with sufficient variance")
    
    return df


def normalize_data(
    df: pd.DataFrame,
    train_mask: pd.Series,
    cols: List[str],
    control_col: str
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame, str]:
    """
    Normalize specified columns using training statistics only.
    
    Args:
        df: Full dataframe
        train_mask: Boolean mask identifying training rows
        cols: Columns to normalize (proxy_A, proxy_B)
        control_col: Control market column name
        
    Returns:
        Tuple of:
            - normalization_params: Dict mapping col -> (mu, sigma)
            - normalized_df: DataFrame with normalized columns
            - actual_control_col: Name of control column used (may be fallback)
    """
    train_df = df[train_mask]
    norm_params = {}
    normalized_df = df.copy()
    
    # Check if control market has sufficient variance in training window
    control_std = train_df[control_col].std()
    
    if control_std < 1e-6:
        warnings.warn(
            f"Control market has insufficient variance (std={control_std:.2e}) in training window. "
            f"Using fallback: (proxy_A + proxy_B) / 2 as control regressor.",
            RuntimeWarning
        )
        
        # Create fallback control
        fallback_col = "control_fallback"
        normalized_df[fallback_col] = (df["proxy_A"] + df["proxy_B"]) / 2
        actual_control_col = fallback_col
        
        # Normalize fallback using training stats
        mu_ctrl = train_df["proxy_A"].add(train_df["proxy_B"]).div(2).mean()
        sigma_ctrl = train_df["proxy_A"].add(train_df["proxy_B"]).div(2).std()
    else:
        actual_control_col = control_col
        mu_ctrl = train_df[control_col].mean()
        sigma_ctrl = train_df[control_col].std()
    
    norm_params[actual_control_col] = (mu_ctrl, sigma_ctrl)
    normalized_df[actual_control_col + "_norm"] = (
        normalized_df[actual_control_col] - mu_ctrl
    ) / sigma_ctrl
    
    # Normalize proxy columns
    for col in cols:
        mu = train_df[col].mean()
        sigma = train_df[col].std()
        
        if sigma < 1e-6:
            raise ValueError(
                f"Column '{col}' has zero variance in training window. Cannot normalize."
            )
        
        norm_params[col] = (mu, sigma)
        normalized_df[col + "_norm"] = (normalized_df[col] - mu) / sigma
    
    return norm_params, normalized_df, actual_control_col


def generate_backtest_folds(
    df_national: pd.DataFrame,
    config: Dict
) -> List[Dict]:
    """
    Generate expanding-window walk-forward backtest folds (pre-launch only).
    
    Args:
        df_national: Validated national dataframe
        config: Configuration dictionary
        
    Returns:
        List of fold dictionaries, each containing:
            - fold_id: int
            - train_start: date
            - train_end: date
            - n_train: int
            - forecast_dates: DatetimeIndex (length = horizon)
    """
    launch_date = pd.Timestamp(config["sku1_launch"])
    min_train = config["backtest_min_train"]
    horizon = config["backtest_horizon"]
    step = config["backtest_step"]
    
    # Only use pre-launch data
    pre_launch_df = df_national[df_national["date"] < launch_date].copy()
    pre_launch_df = pre_launch_df.reset_index(drop=True)
    
    if len(pre_launch_df) < min_train + horizon:
        raise ValueError(
            f"Insufficient pre-launch data for backtesting. "
            f"Need at least {min_train + horizon} months, got {len(pre_launch_df)}"
        )
    
    folds = []
    train_end_idx = min_train - 1
    
    while True:
        train_end = pre_launch_df.iloc[train_end_idx]["date"]
        forecast_start = train_end + pd.DateOffset(months=1)
        forecast_end = train_end + pd.DateOffset(months=horizon)
        
        # Stop if any forecast month is on or after launch
        if forecast_end >= launch_date:
            break
        
        # Ensure all forecast months exist in data
        forecast_dates = pd.date_range(forecast_start, periods=horizon, freq="MS")
        
        # Check if all forecast dates are in pre_launch_df
        missing_forecasts = set(forecast_dates) - set(pre_launch_df["date"])
        if missing_forecasts:
            warnings.warn(
                f"Skipping fold at train_end={train_end.date()}: "
                f"missing forecast dates {missing_forecasts}"
            )
            train_end_idx += step
            continue
        
        folds.append({
            "fold_id": len(folds) + 1,
            "train_start": pre_launch_df.iloc[0]["date"],
            "train_end": train_end,
            "n_train": train_end_idx + 1,
            "forecast_dates": forecast_dates
        })
        
        train_end_idx += step
    
    print(f"✓ Generated {len(folds)} backtest folds (expanding window, pre-launch only)")
    print(f"  First fold train period: {folds[0]['train_start'].date()} to {folds[0]['train_end'].date()}")
    print(f"  Last fold train period: {folds[-1]['train_start'].date()} to {folds[-1]['train_end'].date()}")
    
    return folds


def prepare_monthly_data(
    df: pd.DataFrame,
    analysis_month: pd.Timestamp,
    training_years: int,
    proxy_cols: List[str],
    control_col: str,
    n_seasons: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, int]:
    """
    Prepare normalized training and test data for a single analysis month.
    
    Args:
        df: Full dataframe
        analysis_month: Month to analyze
        training_years: Number of years for training window
        proxy_cols: List of proxy column names
        control_col: Control market column name
        n_seasons: Number of seasons (default 12 for monthly)
        
    Returns:
        Tuple of:
            - Y_train: (n_train, 2) array of normalized proxies
            - X_train: (n_train, 1) array of normalized control
            - X_test: (1, 1) array of normalized control for analysis month
            - norm_params: Dict of normalization parameters
            - n_trimmed: Number of rows after trimming
            
    Raises:
        ValueError: If insufficient training data or validation fails
    """
    # Slice training window
    train_end = analysis_month - pd.DateOffset(months=1)
    train_start = train_end - pd.DateOffset(years=training_years)
    
    train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
    train_df = df[train_mask].copy()
    
    # Check minimum training size
    if len(train_df) < 18:
        raise ValueError(
            f"Insufficient training data for {analysis_month.date()}: "
            f"{len(train_df)} months (need >= 18)"
        )
    
    # Normalize data
    norm_params, normalized_df, actual_control_col = normalize_data(
        df, train_mask, proxy_cols, control_col
    )
    
    # Extract normalized training data
    train_norm_df = normalized_df[train_mask].reset_index(drop=True)
    
    # Trim training rows to avoid KFAS seasonal boundary errors
    n_train_raw = len(train_norm_df)
    n_trimmed = (np.floor((n_train_raw + 1) / n_seasons) * n_seasons - 1).astype(int)
    
    if n_trimmed < 18:
        raise ValueError(
            f"After trimming for seasonal boundary: {n_trimmed} rows (need >= 18). "
            f"Original: {n_train_raw} months."
        )
    
    # Trim from beginning (oldest rows)
    rows_to_drop = n_train_raw - n_trimmed
    train_norm_df = train_norm_df.iloc[rows_to_drop:].reset_index(drop=True)
    
    # Build Y_train (n_train, 2)
    Y_train = train_norm_df[[col + "_norm" for col in proxy_cols]].values
    
    # Build X_train (n_train, 1)
    X_train = train_norm_df[actual_control_col + "_norm"].values.reshape(-1, 1)
    
    # Build X_test (1, 1) for analysis month
    test_mask = normalized_df["date"] == analysis_month
    if test_mask.sum() == 0:
        raise ValueError(f"Analysis month {analysis_month.date()} not found in data")
    
    X_test = normalized_df.loc[test_mask, actual_control_col + "_norm"].values.reshape(-1, 1)
    
    return Y_train, X_train, X_test, norm_params, n_trimmed


def prepare_backtest_fold_data(
    df: pd.DataFrame,
    fold: Dict,
    proxy_cols: List[str],
    control_col: str,
    n_seasons: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, int]:
    """
    Prepare normalized data for a single backtest fold.
    
    Args:
        df: Full dataframe
        fold: Fold dictionary from generate_backtest_folds
        proxy_cols: List of proxy column names
        control_col: Control market column name
        n_seasons: Number of seasons
        
    Returns:
        Tuple of:
            - Y_train: (n_train, 2) array
            - X_train: (n_train, 1) array
            - X_test: (horizon, 1) array  # Note: multi-step horizon
            - norm_params: Dict
            - n_trimmed: int
    """
    train_mask = (df["date"] >= fold["train_start"]) & (df["date"] <= fold["train_end"])
    train_df = df[train_mask].copy()
    
    # Normalize
    norm_params, normalized_df, actual_control_col = normalize_data(
        df, train_mask, proxy_cols, control_col
    )
    
    train_norm_df = normalized_df[train_mask].reset_index(drop=True)
    
    # Trim for seasonal boundary
    n_train_raw = len(train_norm_df)
    n_trimmed = (np.floor((n_train_raw + 1) / n_seasons) * n_seasons - 1).astype(int)
    
    if n_trimmed < 18:
        raise ValueError(
            f"Fold {fold['fold_id']}: insufficient data after trim ({n_trimmed} < 18)"
        )
    
    rows_to_drop = n_train_raw - n_trimmed
    train_norm_df = train_norm_df.iloc[rows_to_drop:].reset_index(drop=True)
    
    Y_train = train_norm_df[[col + "_norm" for col in proxy_cols]].values
    X_train = train_norm_df[actual_control_col + "_norm"].values.reshape(-1, 1)
    
    # X_test for all forecast_dates (multi-step)
    test_mask = normalized_df["date"].isin(fold["forecast_dates"])
    X_test = normalized_df.loc[test_mask, actual_control_col + "_norm"].values.reshape(-1, 1)
    
    if len(X_test) != len(fold["forecast_dates"]):
        raise ValueError(
            f"Fold {fold['fold_id']}: mismatch in forecast dates. "
            f"Expected {len(fold['forecast_dates'])}, got {len(X_test)}"
        )
    
    return Y_train, X_train, X_test, norm_params, n_trimmed

"""
inference.py

De-normalization, delta computation, and MiNT reconciliation for innovation impact measurement.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.covariance import LedoitWolf
import warnings


def denormalize_posterior(
    pred_dist: np.ndarray,
    norm_params: Dict,
    proxy_cols: list
) -> np.ndarray:
    """
    De-normalize posterior predictive draws.
    
    Args:
        pred_dist: Posterior draws, shape (draws, 2) or (draws, 2, steps)
        norm_params: Dict mapping column -> (mu, sigma)
        proxy_cols: List of proxy column names in order
        
    Returns:
        De-normalized posterior draws (same shape as input)
    """
    denorm_dist = pred_dist.copy()
    
    for i, col in enumerate(proxy_cols):
        mu, sigma = norm_params[col]
        
        if len(pred_dist.shape) == 2:
            # Single-step: (draws, 2)
            denorm_dist[:, i] = pred_dist[:, i] * sigma + mu
        elif len(pred_dist.shape) == 3:
            # Multi-step: (draws, 2, steps)
            denorm_dist[:, i, :] = pred_dist[:, i, :] * sigma + mu
        else:
            raise ValueError(f"Unexpected pred_dist shape: {pred_dist.shape}")
    
    return denorm_dist


def denormalize_fitted(
    fitted_vals: np.ndarray,
    norm_params: Dict,
    proxy_cols: list
) -> np.ndarray:
    """
    De-normalize fitted values.
    
    Args:
        fitted_vals: Fitted values array (n_train, 2)
        norm_params: Dict mapping column -> (mu, sigma)
        proxy_cols: List of proxy column names
        
    Returns:
        De-normalized fitted values (n_train, 2)
    """
    denorm_fitted = fitted_vals.copy()
    
    for i, col in enumerate(proxy_cols):
        mu, sigma = norm_params[col]
        denorm_fitted[:, i] = fitted_vals[:, i] * sigma + mu
    
    return denorm_fitted


def compute_delta_global(
    v_cf_draws: np.ndarray,
    actual: float
) -> Tuple[float, float, float, float]:
    """
    Compute global incremental delta from counterfactual draws.
    
    Args:
        v_cf_draws: De-normalized counterfactual posterior draws (draws, 2)
        actual: Actual total portfolio sales for the period
        
    Returns:
        Tuple of (delta_mean, delta_lower, delta_upper, prob_positive)
    """
    # Sum across proxy groups to get total counterfactual per draw
    total_cf_draws = v_cf_draws.sum(axis=1)
    
    # Delta = Actual - Counterfactual
    delta_draws = actual - total_cf_draws
    
    # Summary statistics
    delta_mean = np.mean(delta_draws)
    delta_lower = np.percentile(delta_draws, 10)  # 80% CI
    delta_upper = np.percentile(delta_draws, 90)
    prob_positive = np.mean(delta_draws > 0)
    
    return delta_mean, delta_lower, delta_upper, prob_positive


def compute_ledoit_wolf_weights(
    fitted_vals: np.ndarray,
    y_train_actual: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MiNT reconciliation weights using Ledoit-Wolf shrinkage.
    
    Args:
        fitted_vals: De-normalized fitted values (n_train, 2)
        y_train_actual: De-normalized actual Y_train (n_train, 2)
        
    Returns:
        Tuple of (w, W) where:
            - w: Weight vector (2,) for reconciliation
            - W: Shrunk covariance matrix (2, 2)
    """
    # Compute residuals
    residuals = y_train_actual - fitted_vals
    
    if residuals.shape[0] < 2:
        raise ValueError(
            f"Insufficient residuals for covariance estimation: {residuals.shape[0]} rows"
        )
    
    # Ledoit-Wolf shrinkage estimator
    lw = LedoitWolf()
    lw.fit(residuals)
    W = lw.covariance_
    
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(W)
    if np.any(eigvals <= 0):
        warnings.warn(
            f"Covariance matrix not positive definite (min eigval={eigvals.min():.2e}). "
            f"Adding regularization.",
            RuntimeWarning
        )
        W = W + np.eye(W.shape[0]) * 1e-6
    
    # Compute weights: w = inv(W) @ 1 / (1' @ inv(W) @ 1)
    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        warnings.warn("Covariance matrix singular. Using pseudo-inverse.", RuntimeWarning)
        W_inv = np.linalg.pinv(W)
    
    ones = np.ones(W.shape[0])
    w_unnorm = W_inv @ ones
    w = w_unnorm / w_unnorm.sum()
    
    return w, W


def mint_reconcile(
    delta_mean: float,
    w: np.ndarray,
    inno_names: list
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Reconcile global delta to SKU level using MiNT weights.
    
    Args:
        delta_mean: Global incremental delta
        w: MiNT weight vector (2,)
        inno_names: List of innovation SKU names
        
    Returns:
        Tuple of (delta_inno, shares, coherence_err) where:
            - delta_inno: SKU-level deltas (2,)
            - shares: SKU shares of total delta (2,)
            - coherence_err: Absolute reconciliation error
    """
    if len(w) != len(inno_names):
        raise ValueError(
            f"Weight vector length ({len(w)}) != number of innovations ({len(inno_names)})"
        )
    
    # Allocate global delta proportional to weights
    delta_inno = w * delta_mean
    
    # Compute shares
    if abs(delta_mean) < 1e-6:
        shares = w  # Use weights directly if delta is ~zero
    else:
        shares = delta_inno / delta_mean
    
    # Check coherence
    coherence_err = abs(delta_inno.sum() - delta_mean)
    
    if coherence_err >= 0.01:
        raise ValueError(
            f"MiNT coherence constraint violated: sum(delta_inno) = {delta_inno.sum():.4f}, "
            f"delta_mean = {delta_mean:.4f}, error = {coherence_err:.4f}"
        )
    
    return delta_inno, shares, coherence_err


def evaluate_backtest_fold(
    pred_dist: np.ndarray,
    actual_vals: np.ndarray,
    norm_params: Dict,
    proxy_cols: list,
    ci_level: float = 0.80
) -> Dict:
    """
    Evaluate backtest predictions against actuals for one fold.
    
    Args:
        pred_dist: Posterior draws (draws, 2, steps)
        actual_vals: Actual values (steps, 2) - de-normalized
        norm_params: Normalization parameters (for metadata only)
        proxy_cols: Proxy column names
        ci_level: Credible interval level (default 0.80)
        
    Returns:
        Dict containing per-step per-group metrics:
            - mae, mape, rmse, coverage, pred_mean, pred_lower, pred_upper
    """
    steps = pred_dist.shape[2]
    n_groups = pred_dist.shape[1]
    
    if actual_vals.shape != (steps, n_groups):
        raise ValueError(
            f"Shape mismatch: pred_dist is (*, {n_groups}, {steps}), "
            f"actual_vals is {actual_vals.shape}"
        )
    
    alpha = 1 - ci_level
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    
    results = []
    
    for h in range(steps):
        for g in range(n_groups):
            actual = actual_vals[h, g]
            draws = pred_dist[:, g, h]
            
            pred_mean = np.mean(draws)
            pred_lower = np.percentile(draws, lower_pct)
            pred_upper = np.percentile(draws, upper_pct)
            
            # Metrics
            mae = abs(actual - pred_mean)
            mape = (mae / abs(actual)) * 100 if abs(actual) > 1e-6 else np.inf
            rmse = np.sqrt((actual - pred_mean) ** 2)
            coverage = 1 if pred_lower <= actual <= pred_upper else 0
            
            results.append({
                "horizon_step": h + 1,
                "group": proxy_cols[g],
                "actual": actual,
                "pred_mean": pred_mean,
                "pred_lower": pred_lower,
                "pred_upper": pred_upper,
                "mae": mae,
                "mape": mape,
                "rmse": rmse,
                "coverage": coverage
            })
    
    return results


def compute_oos_r2(
    actual_vals: np.ndarray,
    pred_means: np.ndarray
) -> Tuple[float, float]:
    """
    Compute out-of-sample R² for each proxy group across forecast steps.
    
    Args:
        actual_vals: Actual values (steps, 2)
        pred_means: Predicted means (steps, 2)
        
    Returns:
        Tuple of (r2_A, r2_B)
    """
    n_groups = actual_vals.shape[1]
    r2_values = []
    
    for g in range(n_groups):
        actuals = actual_vals[:, g]
        preds = pred_means[:, g]
        
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else -np.inf
        r2_values.append(r2)
    
    return tuple(r2_values)


def validate_r2(group_r2: np.ndarray, context: str) -> bool:
    """
    Validate R² values and print warnings/errors.
    
    Args:
        group_r2: Array of R² values per group
        context: Context string for error messages (e.g., "2023-09-01")
        
    Returns:
        True if all R² > 0.70, False otherwise
    """
    all_ok = True
    
    for i, r2 in enumerate(group_r2):
        if r2 < 0.00:
            print(f"  ⚠ CRITICAL [{context}] Group {i+1}: R²={r2:.4f} < 0.00 (worse than naive mean)")
            warnings.warn(
                f"[{context}] Group {i+1} R²={r2:.4f} < 0.00 - model worse than naive prediction",
                RuntimeWarning
            )
            all_ok = False
        elif r2 < 0.70:
            print(f"  ⚠ WARNING [{context}] Group {i+1}: R²={r2:.4f} < 0.70 (insufficient accuracy)")
            all_ok = False
        elif r2 < 0.85:
            print(f"  ✓ [{context}] Group {i+1}: R²={r2:.4f} (acceptable)")
        else:
            print(f"  ✓ [{context}] Group {i+1}: R²={r2:.4f} (good)")
    
    return all_ok

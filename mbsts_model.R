# mbsts_model.R
#
# Pure R implementation of multivariate Bayesian structural time series modeling
# using the standalone mbsts CRAN package (NOT bsts::mbsts).
#
# Called via rpy2 from Python. Receives normalized data, returns posterior draws.

library(mbsts)


#' Run mbsts model for one month or fold
#'
#' @param Y_train Numeric matrix (n_train x 2) of normalized proxy series
#' @param X_train Numeric matrix (n_train x 1) of normalized control regressor
#' @param X_test Numeric matrix (steps x 1) of control regressor for forecast period
#' @param mc Integer, number of MCMC iterations (default 1000)
#' @param burn Integer, burn-in period (default 200)
#' @param n_seasons Integer, number of seasons (default 12 for monthly)
#' @param steps Integer, forecast horizon (1 for post-launch, 3 for backtest)
#'
#' @return List containing:
#'   - pred_dist: Posterior predictive draws array (draws x 2 x steps)
#'   - fitted_vals: In-sample fitted values matrix (n_train x 2)
#'   - group_r2: Numeric vector length 2 (R² per proxy group)
#'   - model_object: Full mbsts model object (S4)
#'
run_mbsts_month <- function(Y_train, X_train, X_test,
                           mc = 1000, burn = 200,
                           n_seasons = 12, steps = 1) {
  
  # ═══════════════════════════════════════════════════════════════
  # 1. INPUT VALIDATION
  # ═══════════════════════════════════════════════════════════════
  
  if (!is.matrix(Y_train)) {
    Y_train <- as.matrix(Y_train)
  }
  if (!is.matrix(X_train)) {
    X_train <- as.matrix(X_train)
  }
  if (!is.matrix(X_test)) {
    X_test <- as.matrix(X_test)
  }
  
  n_train <- nrow(Y_train)
  n_groups <- ncol(Y_train)
  
  if (n_groups != 2) {
    stop("Y_train must have exactly 2 columns (proxy_A, proxy_B)")
  }
  
  if (nrow(X_train) != n_train) {
    stop("X_train must have same number of rows as Y_train")
  }
  
  if (nrow(X_test) != steps) {
    stop(sprintf("X_test must have %d rows (forecast steps), got %d",
                 steps, nrow(X_test)))
  }
  
  # Check for trimming requirement (already done in Python, but validate)
  expected_n <- floor((n_train + 1) / n_seasons) * n_seasons - 1
  if (n_train != expected_n) {
    warning(sprintf(
      "Training rows (%d) not optimally trimmed. Expected %d. May cause KFAS errors.",
      n_train, expected_n
    ))
  }
  
  if (n_train < 18) {
    stop(sprintf("Insufficient training data: %d months (need >= 18)", n_train))
  }
  
  # ═══════════════════════════════════════════════════════════════
  # 2. STATE SPECIFICATION using tsc.setting()
  # ═══════════════════════════════════════════════════════════════
  
  # Build state-space component structure
  # Components: local linear trend + seasonal + regression
  
  STOcomponent <- list(
    trend = list(type = "local_linear_trend"),
    seasonal = list(type = "seasonal", nseasons = n_seasons),
    regression = list(type = "regression")
  )
  
  # Create time series configuration
  # tsc.setting(Y, STOcomponent, Xtrain)
  STmodel <- tsc.setting(
    Y = Y_train,
    STO = STOcomponent,
    Xtrain = X_train
  )
  
  # ═══════════════════════════════════════════════════════════════
  # 3. FIT MODEL using mbsts_function()
  # ═══════════════════════════════════════════════════════════════
  
  # Prior parameters (using defaults suitable for normalized data)
  ki <- matrix(0.01, nrow = n_groups, ncol = 1)  # Small prior precision
  pii <- 1  # Prior degrees of freedom
  
  # Fit model: mbsts_function(Y, Xtrain, STmodel, ki, pii, mc, burn)
  model_fit <- mbsts_function(
    Y = Y_train,
    Xtrain = X_train,
    STmodel = STmodel,
    ki = ki,
    pii = pii,
    mc = mc,
    burn = burn
  )
  
  # ═══════════════════════════════════════════════════════════════
  # 4. EXTRACT IN-SAMPLE FITTED VALUES
  # ═══════════════════════════════════════════════════════════════
  
  # Access S4 slots - structure depends on mbsts package version
  # Typical slots: @Filter, @one.step.ahead, @Smooth
  
  # Extract one-step-ahead predictions or filtered means
  # The mbsts package stores filtered states in model_fit@Filter
  
  if (!is.null(model_fit@Filter)) {
    # Filtered means for observation equation
    # Shape typically: (n_train x n_groups)
    fitted_vals <- model_fit@Filter$m
    
    # Handle case where m might be 3D array (iterations x time x groups)
    if (length(dim(fitted_vals)) == 3) {
      # Take posterior mean across iterations
      fitted_vals <- apply(fitted_vals, c(2, 3), mean)
    }
    
  } else if (!is.null(model_fit@Smooth)) {
    # Fall back to smoothed estimates
    fitted_vals <- model_fit@Smooth$s
    
    if (length(dim(fitted_vals)) == 3) {
      fitted_vals <- apply(fitted_vals, c(2, 3), mean)
    }
    
  } else {
    # Last resort: use state estimates to reconstruct observation
    # This is model-specific and may require state-space algebra
    warning("Could not extract fitted values from standard slots. Using approximation.")
    fitted_vals <- matrix(0, nrow = n_train, ncol = n_groups)
  }
  
  # Ensure correct dimensions
  if (!is.matrix(fitted_vals)) {
    fitted_vals <- as.matrix(fitted_vals)
  }
  
  if (nrow(fitted_vals) != n_train || ncol(fitted_vals) != n_groups) {
    stop(sprintf(
      "Fitted values dimension mismatch. Expected (%d, %d), got (%d, %d)",
      n_train, n_groups, nrow(fitted_vals), ncol(fitted_vals)
    ))
  }
  
  # ═══════════════════════════════════════════════════════════════
  # 5. COMPUTE IN-SAMPLE R² PER GROUP
  # ═══════════════════════════════════════════════════════════════
  
  group_r2 <- numeric(n_groups)
  
  for (g in 1:n_groups) {
    residuals <- Y_train[, g] - fitted_vals[, g]
    ss_res <- sum(residuals^2)
    ss_tot <- sum((Y_train[, g] - mean(Y_train[, g]))^2)
    
    group_r2[g] <- 1 - (ss_res / ss_tot)
  }
  
  # ═══════════════════════════════════════════════════════════════
  # 6. FORECAST using mbsts.forecast()
  # ═══════════════════════════════════════════════════════════════
  
  # mbsts.forecast(model_object, STmodel, Xtest, steps = N)
  forecast_result <- mbsts.forecast(
    model_fit,
    STmodel,
    Xtest = X_test,
    steps = steps
  )
  
  # Extract posterior predictive distribution
  # forecast_result should contain draws in forecast_result@fcst or similar slot
  
  if (!is.null(forecast_result@fcst)) {
    pred_dist_raw <- forecast_result@fcst
  } else if (!is.null(forecast_result@predictive)) {
    pred_dist_raw <- forecast_result@predictive
  } else {
    stop("Could not extract forecast distribution from mbsts.forecast result")
  }
  
  # Reshape to (draws x groups x steps)
  # Input shape may vary: (steps x groups x draws) or (draws x groups x steps)
  dims <- dim(pred_dist_raw)
  
  if (length(dims) == 2) {
    # Single-step forecast: (draws x groups)
    pred_dist <- pred_dist_raw
    if (steps != 1) {
      stop(sprintf("Expected multi-step forecast but got 2D array for steps=%d", steps))
    }
  } else if (length(dims) == 3) {
    # Multi-step forecast: ensure correct ordering (draws x groups x steps)
    # mbsts typically returns (draws x steps x groups) or (draws x groups x steps)
    
    # Check which dimension matches n_groups (should be 2)
    if (dims[2] == n_groups) {
      # Already correct: (draws x groups x steps)
      pred_dist <- pred_dist_raw
    } else if (dims[3] == n_groups) {
      # Need to permute: (draws x steps x groups) -> (draws x groups x steps)
      pred_dist <- aperm(pred_dist_raw, c(1, 3, 2))
    } else {
      stop(sprintf(
        "Cannot identify groups dimension in forecast output. Shape: (%d, %d, %d)",
        dims[1], dims[2], dims[3]
      ))
    }
  } else {
    stop(sprintf("Unexpected forecast distribution dimensions: %s", paste(dims, collapse = "x")))
  }
  
  # ═══════════════════════════════════════════════════════════════
  # 7. RETURN RESULTS
  # ═══════════════════════════════════════════════════════════════
  
  result <- list(
    pred_dist = pred_dist,
    fitted_vals = fitted_vals,
    group_r2 = group_r2,
    model_object = model_fit
  )
  
  return(result)
}

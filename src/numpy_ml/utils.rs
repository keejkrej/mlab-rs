//! Shared utilities for `numpy_ml` models.

/// Stable log-sum-exp of a slice of log probabilities.
pub fn logsumexp(log_probs: &[f64]) -> f64 {
    let max_val = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_probs.iter().map(|&p| (p - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

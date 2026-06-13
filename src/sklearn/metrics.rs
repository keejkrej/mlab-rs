use ndarray::Array1;

/// Compute accuracy score for classification.
pub fn accuracy_score<T: PartialEq>(y_true: &Array1<T>, y_pred: &Array1<T>) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mut correct = 0;
    for i in 0..n {
        if y_true[i] == y_pred[i] {
            correct += 1;
        }
    }
    (correct as f64) / (n as f64)
}

/// Compute mean squared error for regression.
pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        sum_sq += (y_true[i] - y_pred[i]).powi(2);
    }
    sum_sq / (n as f64)
}

/// Compute R^2 (coefficient of determination) regression score.
pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mean_y = y_true.mean().unwrap_or(0.0);
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..n {
        ss_tot += (y_true[i] - mean_y).powi(2);
        ss_res += (y_true[i] - y_pred[i]).powi(2);
    }
    if ss_tot < 1e-14 {
        return 0.0;
    }
    1.0 - (ss_res / ss_tot)
}

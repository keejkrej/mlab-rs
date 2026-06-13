use ndarray::{Array1, Array2};

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

/// Compute the confusion matrix for classification labels.
pub fn confusion_matrix(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Array2<usize> {
    let mut labels = y_true.iter().copied().chain(y_pred.iter().copied()).collect::<Vec<_>>();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    let n = labels.len();
    let mut matrix = Array2::zeros((n, n));
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        let i = labels.iter().position(|&label| label == *t).unwrap();
        let j = labels.iter().position(|&label| label == *p).unwrap();
        matrix[[i, j]] += 1;
    }
    matrix
}

#[derive(Debug)]
pub struct ClassificationReportEntry {
    pub label: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

/// Compute precision, recall, and f1-score for each class.
pub fn classification_report(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Vec<ClassificationReportEntry> {
    let mut labels = y_true.iter().copied().chain(y_pred.iter().copied()).collect::<Vec<_>>();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    labels
        .into_iter()
        .map(|label| {
            let tp = y_true.iter().zip(y_pred.iter()).filter(|(t, p)| **t == label && **p == label).count() as f64;
            let fp = y_true.iter().zip(y_pred.iter()).filter(|(t, p)| **t != label && **p == label).count() as f64;
            let fn_count = y_true.iter().zip(y_pred.iter()).filter(|(t, p)| **t == label && **p != label).count() as f64;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            ClassificationReportEntry { label, precision, recall, f1_score: f1 }
        })
        .collect()
}

/// Compute R^2 (coefficient of determination) regression score.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_confusion_matrix_and_report() {
        let y_true = array![0.0, 1.0, 1.0, 0.0];
        let y_pred = array![0.0, 0.0, 1.0, 1.0];
        let cm = confusion_matrix(&y_true, &y_pred);
        assert_eq!(cm.dim(), (2, 2));
        assert_eq!(cm[[0, 0]], 1);
        assert_eq!(cm[[1, 1]], 1);

        let report = classification_report(&y_true, &y_pred);
        assert_eq!(report.len(), 2);
        assert!(report[0].precision >= 0.0);
    }
}

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

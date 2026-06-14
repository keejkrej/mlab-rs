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

fn confusion_matrix_binary(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tn = 0.0;
    let mut fn_ = 0.0;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        match (*t == 1.0, *p == 1.0) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (false, false) => tn += 1.0,
            (true, false) => fn_ += 1.0,
        }
    }
    (tp, fp, tn, fn_)
}

fn _binary_clf_curve(y_true: &[f64], y_score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y_true.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| y_score[j].partial_cmp(&y_score[i]).unwrap_or(std::cmp::Ordering::Equal));

    let mut fps = Vec::new();
    let mut tps = Vec::new();
    let mut thresholds = Vec::new();

    let mut tp_accum = 0.0;
    let mut fp_accum = 0.0;
    let n_pos = y_true.iter().filter(|&&v| v == 1.0).count() as f64;
    let n_neg = y_true.iter().filter(|&&v| v == 0.0).count() as f64;

    let mut prev_score = f64::INFINITY;
    for &idx in &indices {
        let score = y_score[idx];
        if score != prev_score {
            thresholds.push(score);
            fps.push(fp_accum);
            tps.push(tp_accum);
            prev_score = score;
        }
        if y_true[idx] == 1.0 {
            tp_accum += 1.0;
        } else {
            fp_accum += 1.0;
        }
    }
    thresholds.push(f64::NEG_INFINITY);
    fps.push(fp_accum);
    tps.push(tp_accum);

    let fps: Vec<f64> = fps.iter().map(|&f| f / n_neg).collect();
    let tps: Vec<f64> = tps.iter().map(|&t| t / n_pos).collect();

    (fps, tps, thresholds)
}

pub fn precision_score(y_true: &[f64], y_pred: &[f64], average: &str) -> f64 {
    match average {
        "binary" => {
            let (tp, fp, _, _) = confusion_matrix_binary(y_true, y_pred);
            if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) }
        }
        "micro" => {
            let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
            labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            labels.dedup();
            let mut total_tp = 0.0;
            let mut total_fp = 0.0;
            for &label in &labels {
                let (tp, fp, _, _) = confusion_matrix_binary(
                    &y_true.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                    &y_pred.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                );
                total_tp += tp;
                total_fp += fp;
            }
            if total_tp + total_fp == 0.0 { 0.0 } else { total_tp / (total_tp + total_fp) }
        }
        "macro" => {
            let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
            labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            labels.dedup();
            let mut sum = 0.0;
            for &label in &labels {
                let (tp, fp, _, _) = confusion_matrix_binary(
                    &y_true.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                    &y_pred.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                );
                sum += if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
            }
            sum / labels.len() as f64
        }
        _ => panic!("Unknown average type: {}", average),
    }
}

pub fn recall_score(y_true: &[f64], y_pred: &[f64], average: &str) -> f64 {
    match average {
        "binary" => {
            let (tp, _, _, fn_) = confusion_matrix_binary(y_true, y_pred);
            if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) }
        }
        "micro" => {
            let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
            labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            labels.dedup();
            let mut total_tp = 0.0;
            let mut total_fn = 0.0;
            for &label in &labels {
                let (tp, _, _, fn_) = confusion_matrix_binary(
                    &y_true.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                    &y_pred.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                );
                total_tp += tp;
                total_fn += fn_;
            }
            if total_tp + total_fn == 0.0 { 0.0 } else { total_tp / (total_tp + total_fn) }
        }
        "macro" => {
            let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
            labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            labels.dedup();
            let mut sum = 0.0;
            for &label in &labels {
                let (tp, _, _, fn_) = confusion_matrix_binary(
                    &y_true.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                    &y_pred.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
                );
                sum += if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
            }
            sum / labels.len() as f64
        }
        _ => panic!("Unknown average type: {}", average),
    }
}

pub fn f1_score(y_true: &[f64], y_pred: &[f64], average: &str) -> f64 {
    let p = precision_score(y_true, y_pred, average);
    let r = recall_score(y_true, y_pred, average);
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

pub fn roc_curve(y_true: &[f64], y_score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    _binary_clf_curve(y_true, y_score)
}

pub fn roc_auc_score(y_true: &[f64], y_score: &[f64]) -> f64 {
    let (fpr, tpr, _) = roc_curve(y_true, y_score);
    let mut auc = 0.0;
    for i in 1..fpr.len() {
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
    }
    auc
}

pub fn precision_recall_curve(y_true: &[f64], y_score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (fps, tps, thresholds) = _binary_clf_curve(y_true, y_score);
    let n = thresholds.len();
    let mut precision = Vec::with_capacity(n);
    let mut recall = Vec::with_capacity(n);

    for i in 0..n {
        let p = tps[i] + fps[i];
        precision.push(if p == 0.0 { 1.0 } else { tps[i] / p });
        recall.push(tps[i]);
    }

    (precision, recall, thresholds)
}

pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).abs()).sum();
    sum / n as f64
}

pub fn log_loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let eps = 1e-15;
    let n = y_true.len() as f64;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let p_clamped = p.max(eps).min(1.0 - eps);
            t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln()
        })
        .sum();
    -sum / n
}

pub fn balanced_accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mut labels: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    let mut sum = 0.0;
    for &label in &labels {
        let (tp, _, _, fn_) = confusion_matrix_binary(
            &y_true.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
            &y_pred.iter().map(|&v| if v == label { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
        );
        sum += if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
    }
    sum / labels.len() as f64
}

pub fn matthews_corrcoef(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let (tp, fp, tn, fn_) = confusion_matrix_binary(y_true, y_pred);
    let numerator = tp * tn - fp * fn_;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

pub fn cohen_kappa_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let (tp, fp, tn, fn_) = confusion_matrix_binary(y_true, y_pred);
    let p_o = (tp + tn) / n;
    let n_pos = tp + fn_;
    let n_neg = fp + tn;
    let expected_pos = (n_pos / n) * ((tp + fp) / n);
    let expected_neg = (n_neg / n) * ((fn_ + tn) / n);
    let p_e = expected_pos + expected_neg;
    if (1.0 - p_e).abs() < 1e-14 {
        0.0
    } else {
        (p_o - p_e) / (1.0 - p_e)
    }
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

    #[test]
    fn test_precision_score_binary() {
        let y_true = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 0.0, 1.0, 1.0];
        // TP=2, FP=1 => precision=2/3
        let p = precision_score(&y_true, &y_pred, "binary");
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_score_binary() {
        let y_true = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 0.0, 1.0, 1.0];
        // TP=2, FN=1 => recall=2/3
        let r = recall_score(&y_true, &y_pred, "binary");
        assert!((r - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score_binary() {
        let y_true = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 0.0, 1.0, 1.0];
        let f = f1_score(&y_true, &y_pred, "binary");
        // precision=2/3, recall=2/3, f1=2/3
        assert!((f - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_macro() {
        // Class 0: TP=1, FP=0 => p=1.0; TP=1, FN=1 => r=0.5
        // Class 1: TP=2, FP=1 => p=2/3; TP=2, FN=0 => r=1.0
        // macro_p = (1 + 2/3)/2 = 5/6
        // macro_r = (0.5 + 1.0)/2 = 3/4
        let y_true = [0.0, 1.0, 1.0, 0.0];
        let y_pred = [0.0, 1.0, 1.0, 1.0];
        let p = precision_score(&y_true, &y_pred, "macro");
        let r = recall_score(&y_true, &y_pred, "macro");
        assert!((p - 5.0 / 6.0).abs() < 1e-10);
        assert!((r - 3.0 / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_micro_averaging() {
        let y_true = [0.0, 1.0, 1.0, 0.0];
        let y_pred = [0.0, 1.0, 1.0, 1.0];
        // micro precision == micro recall == micro f1 == accuracy when binary
        let p = precision_score(&y_true, &y_pred, "micro");
        let r = recall_score(&y_true, &y_pred, "micro");
        assert!((p - 3.0 / 4.0).abs() < 1e-10);
        assert!((r - 3.0 / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_curve() {
        let y_true = [0.0, 0.0, 1.0, 1.0];
        let y_score = [0.1, 0.4, 0.35, 0.8];
        let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score);
        assert!(!fpr.is_empty());
        assert_eq!(fpr.len(), tpr.len());
        assert_eq!(fpr.len(), thresholds.len());
        // First point should be (0,0) or close
        assert!((fpr[0] - 0.0).abs() < 1e-10);
        assert!((tpr[0] - 0.0).abs() < 1e-10);
        // Monotonicity
        for i in 1..fpr.len() {
            assert!(fpr[i] >= fpr[i - 1] - 1e-10);
            assert!(tpr[i] >= tpr[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_roc_auc_score() {
        let y_true = [0.0, 0.0, 1.0, 1.0];
        let y_score = [0.1, 0.4, 0.35, 0.8];
        let auc = roc_auc_score(&y_true, &y_score);
        // Perfect ranking would be 1.0; this is 0.75
        assert!((auc - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = [0.0, 0.0, 1.0, 1.0];
        let y_score = [0.1, 0.2, 0.8, 0.9];
        let auc = roc_auc_score(&y_true, &y_score);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_curve() {
        let y_true = [0.0, 0.0, 1.0, 1.0];
        let y_score = [0.1, 0.4, 0.35, 0.8];
        let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score);
        assert!(!precision.is_empty());
        assert_eq!(precision.len(), recall.len());
        assert_eq!(precision.len(), thresholds.len());
        // Last recall should be 1.0 (fully expanded point)
        assert!((recall[recall.len() - 1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_absolute_error() {
        let y_true = [3.0, -0.5, 2.0, 7.0];
        let y_pred = [2.5, 0.0, 2.0, 8.0];
        // |0.5| + |0.5| + |0| + |1| = 2.0 / 4 = 0.5
        let mae = mean_absolute_error(&y_true, &y_pred);
        assert!((mae - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss() {
        let y_true = [1.0, 0.0, 1.0, 0.0];
        let y_pred = [0.9, 0.1, 0.8, 0.3];
        let ll = log_loss(&y_true, &y_pred);
        // Should be positive and reasonable
        assert!(ll > 0.0);
        assert!(ll < 1.0);
    }

    #[test]
    fn test_log_loss_perfect() {
        let y_true = [1.0, 0.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 1.0, 0.0];
        let ll = log_loss(&y_true, &y_pred);
        // With clamping, perfect predictions give ~0
        assert!(ll < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy() {
        // Class 0: TP=1, FN=1 => recall=0.5
        // Class 1: TP=2, FN=0 => recall=1.0
        // balanced = (0.5 + 1.0) / 2 = 0.75
        let y_true = [0.0, 1.0, 1.0, 0.0];
        let y_pred = [0.0, 1.0, 1.0, 1.0];
        let ba = balanced_accuracy_score(&y_true, &y_pred);
        assert!((ba - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_matthews_corrcoef() {
        let y_true = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 0.0, 1.0, 1.0];
        // TP=2, FP=1, TN=1, FN=1
        // MCC = (2*1 - 1*1) / sqrt(3*3*2*2) = 1 / sqrt(36) = 1/6
        let mcc = matthews_corrcoef(&y_true, &y_pred);
        assert!((mcc - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa() {
        let y_true = [1.0, 0.0, 1.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 0.0, 1.0, 1.0];
        let kappa = cohen_kappa_score(&y_true, &y_pred);
        // Should be in [-1, 1]
        assert!(kappa >= -1.0 && kappa <= 1.0);
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        let y_true = [1.0, 0.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 1.0, 0.0];
        let kappa = cohen_kappa_score(&y_true, &y_pred);
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_perfect() {
        let y_true = [1.0, 0.0, 1.0, 0.0];
        let y_pred = [1.0, 0.0, 1.0, 0.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred);
        assert!((mcc - 1.0).abs() < 1e-10);
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

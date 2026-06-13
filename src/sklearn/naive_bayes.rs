use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Gaussian Naive Bayes classifier.
pub struct GaussianNB {
    pub classes: Option<Array1<f64>>,
    pub class_prior: Option<Array1<f64>>,
    pub theta: Option<Array2<f64>>, // class means
    pub var: Option<Array2<f64>>,   // class variances
}

impl GaussianNB {
    /// Create a new Gaussian Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            classes: None,
            class_prior: None,
            theta: None,
            var: None,
        }
    }

    /// Fit Gaussian Naive Bayes according to X, y.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        assert_eq!(nrows, y.len(), "X and y must have same number of samples");

        let mut class_indices = HashMap::new();
        for (i, &val) in y.iter().enumerate() {
            class_indices.entry(val.to_bits()).or_insert_with(Vec::new).push(i);
        }

        let mut sorted_classes: Vec<f64> = class_indices.keys().map(|&b| f64::from_bits(b)).collect();
        sorted_classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n_classes = sorted_classes.len();

        let mut classes = Array1::zeros(n_classes);
        let mut class_prior = Array1::zeros(n_classes);
        let mut theta = Array2::zeros((n_classes, ncols));
        let mut var = Array2::zeros((n_classes, ncols));

        let eps = 1e-9; // variance smoothing

        for (c_idx, &c_val) in sorted_classes.iter().enumerate() {
            classes[c_idx] = c_val;
            let indices = &class_indices[&c_val.to_bits()];
            let n_c = indices.len() as f64;
            class_prior[c_idx] = n_c / (nrows as f64);

            // Compute mean (theta) for this class
            for feat in 0..ncols {
                let mut sum = 0.0;
                for &idx in indices {
                    sum += x[[idx, feat]];
                }
                let mean = sum / n_c;
                theta[[c_idx, feat]] = mean;

                // Compute variance
                let mut var_sum = 0.0;
                for &idx in indices {
                    var_sum += (x[[idx, feat]] - mean).powi(2);
                }
                var[[c_idx, feat]] = (var_sum / n_c) + eps;
            }
        }

        self.classes = Some(classes);
        self.class_prior = Some(class_prior);
        self.theta = Some(theta);
        self.var = Some(var);
    }

    /// Predict class labels for samples in X.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let classes = self.classes.as_ref().expect("GaussianNB not fitted");
        let class_prior = self.class_prior.as_ref().expect("GaussianNB not fitted");
        let theta = self.theta.as_ref().expect("GaussianNB not fitted");
        let var = self.var.as_ref().expect("GaussianNB not fitted");

        let (nrows, ncols) = x.dim();
        let n_classes = classes.len();
        let mut y_pred = Array1::zeros(nrows);

        for r in 0..nrows {
            let mut best_class_idx = 0;
            let mut best_log_prob = f64::NEG_INFINITY;

            for c_idx in 0..n_classes {
                let prior = class_prior[c_idx];
                let mut log_prob = prior.ln();

                for feat in 0..ncols {
                    let t = theta[[c_idx, feat]];
                    let v = var[[c_idx, feat]];
                    let val = x[[r, feat]];

                    let exponent = -((val - t).powi(2)) / (2.0 * v);
                    let normalizer = (2.0 * std::f64::consts::PI * v).sqrt().ln();
                    log_prob += exponent - normalizer;
                }

                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class_idx = c_idx;
                }
            }

            y_pred[r] = classes[best_class_idx];
        }

        y_pred
    }
}

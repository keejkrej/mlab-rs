use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// k-Nearest Neighbors for classification and regression.
pub struct KNN {
    pub k: usize,
    pub classifier: bool,
    pub weights: String,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

impl KNN {
    /// Create a new KNN model.
    pub fn new(k: usize, classifier: bool, weights: &str) -> Self {
        Self {
            k,
            classifier,
            weights: weights.to_string(),
            x: None,
            y: None,
        }
    }

    /// Fit the model to training data.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        assert_eq!(x.nrows(), y.len(), "X and y must have same number of samples");
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
    }

    /// Predict targets for each row in `x`.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let train_x = self.x.as_ref().expect("Model not fitted");
        let train_y = self.y.as_ref().expect("Model not fitted");

        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let query = x.row(i).to_owned();
            let nearest = self.nearest_neighbors(&query, train_x, train_y);
            predictions[i] = self.aggregate(&nearest);
        }
        predictions
    }

    fn nearest_neighbors<'a>(&self, query: &Array1<f64>, train_x: &'a Array2<f64>, train_y: &'a Array1<f64>) -> Vec<(usize, f64, f64)> {
        let mut dists: Vec<(usize, f64, f64)> = (0..train_x.nrows())
            .map(|idx| {
                let d = euclidean(query, &train_x.row(idx).to_owned());
                (idx, d, train_y[idx])
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().take(self.k).collect()
    }

    fn aggregate(&self, nearest: &[(usize, f64, f64)]) -> f64 {
        if self.classifier {
            if self.weights == "uniform" {
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for &(_, _, val) in nearest {
                    *counts.entry(val.to_bits() as i64).or_insert(0) += 1;
                }
                let (best_bits, _) = counts.iter()
                    .fold((0i64, 0usize), |acc, (&k, &v)| if v > acc.1 || (v == acc.1 && k < acc.0) { (k, v) } else { acc });
                f64::from_bits(best_bits as u64)
            } else {
                // distance-weighted: sum inverse-distance weights per class
                let mut scores: HashMap<i64, f64> = HashMap::new();
                for &(_, dist, val) in nearest {
                    let w = if dist == 0.0 { 1e12 } else { 1.0 / dist };
                    *scores.entry(val.to_bits() as i64).or_insert(0.0) += w;
                }
                let (best_bits, _) = scores.iter()
                    .fold((0i64, 0.0), |acc, (&k, &v)| if v > acc.1 { (k, v) } else { acc });
                f64::from_bits(best_bits as u64)
            }
        } else if self.weights == "uniform" {
            nearest.iter().map(|&(_, _, val)| val).sum::<f64>() / nearest.len() as f64
        } else {
            let mut num = 0.0;
            let mut den = 0.0;
            for &(_, dist, val) in nearest {
                let w = if dist == 0.0 { 1e12 } else { 1.0 / dist };
                num += w * val;
                den += w;
            }
            if den == 0.0 { 0.0 } else { num / den }
        }
    }
}

fn euclidean(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// Nadaraya-Watson kernel regression.
pub struct KernelRegression {
    kernel: Box<dyn crate::numpy_ml::utils::kernels::Kernel>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

impl KernelRegression {
    /// Create a new kernel regression model with the given kernel name.
    pub fn new(kernel: Option<&str>) -> Result<Self, String> {
        Ok(Self {
            kernel: crate::numpy_ml::utils::kernels::kernel_initializer(kernel)?,
            x: None,
            y: None,
        })
    }

    /// Fit the regression model.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
    }

    /// Predict targets for each row in `x`.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let train_x = self.x.as_ref().expect("Model not fitted");
        let train_y = self.y.as_ref().expect("Model not fitted");

        let sim = self.kernel.kernel(train_x, Some(x)); // shape (N_train, N_query)
        let mut preds = Array1::zeros(x.nrows());
        for j in 0..x.nrows() {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..train_x.nrows() {
                num += sim[[i, j]] * train_y[i];
                den += sim[[i, j]];
            }
            preds[j] = if den == 0.0 { 0.0 } else { num / den };
        }
        preds
    }
}

/// Gaussian Process regression.
pub struct GPRegression {
    kernel: Box<dyn crate::numpy_ml::utils::kernels::Kernel>,
    alpha: f64,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
    k: Option<Array2<f64>>,
}

impl GPRegression {
    /// Create a new GP regression model.
    pub fn new(kernel: Option<&str>, alpha: f64) -> Result<Self, String> {
        Ok(Self {
            kernel: crate::numpy_ml::utils::kernels::kernel_initializer(kernel)?,
            alpha,
            x: None,
            y: None,
            k: None,
        })
    }

    /// Fit the GP prior to training data.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.k = Some(self.kernel.kernel(x, Some(x)));
        self.x = Some(x.to_owned());
        self.y = Some(y.to_owned());
    }

    /// Predict means and confidence intervals for `x`.
    pub fn predict(&self, x: &Array2<f64>, return_cov: bool) -> Result<(Array1<f64>, Array1<f64>, Option<Array2<f64>>), String> {
        let train_x = self.x.as_ref().ok_or("Model not fitted")?;
        let train_y = self.y.as_ref().ok_or("Model not fitted")?;
        let k = self.k.as_ref().ok_or("Model not fitted")?;

        let n = train_x.nrows();
        let k_star = self.kernel.kernel(x, Some(train_x)); // (N_query, N_train)
        let k_star_star = self.kernel.kernel(x, Some(x)); // (N_query, N_query)

        let mut k_y = k.to_owned();
        for i in 0..n {
            k_y[[i, i]] += self.alpha;
        }
        let k_y_inv = crate::sp::linalg::inv(&k_y)?;

        let pp_mean = k_star.dot(&k_y_inv).dot(train_y);
        let pp_cov = &k_star_star - k_star.dot(&k_y_inv).dot(&k_star.t());

        let conf = (0..x.nrows()).map(|i| 1.96 * pp_cov[[i, i]].sqrt()).collect();

        if return_cov {
            Ok((pp_mean, conf, Some(pp_cov)))
        } else {
            Ok((pp_mean, conf, None))
        }
    }
}

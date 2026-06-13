use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;

// --- preprocessing submodule ---

pub mod preprocessing {
    use super::*;

    /// StandardScaler standardizes features by removing the mean and scaling to unit variance.
    pub struct StandardScaler {
        pub mean: Option<Array1<f64>>,
        pub scale: Option<Array1<f64>>,
    }

    impl StandardScaler {
        pub fn new() -> Self {
            Self {
                mean: None,
                scale: None,
            }
        }

        pub fn fit(&mut self, x: &Array2<f64>) {
            let (nrows, ncols) = x.dim();
            if nrows <= 1 {
                let mean = x.mean_axis(Axis(0)).unwrap();
                let scale = Array1::from_elem(ncols, 1.0);
                self.mean = Some(mean);
                self.scale = Some(scale);
                return;
            }
            let mean = x.mean_axis(Axis(0)).unwrap();
            let mut std_dev = Array1::zeros(ncols);
            for col in 0..ncols {
                let col_mean = mean[col];
                let mut sum_sq = 0.0;
                for row in 0..nrows {
                    sum_sq += (x[[row, col]] - col_mean).powi(2);
                }
                let variance = sum_sq / (nrows as f64);
                let sd = if variance > 1e-14 { variance.sqrt() } else { 1.0 };
                std_dev[col] = sd;
            }
            self.mean = Some(mean);
            self.scale = Some(std_dev);
        }

        pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
            let mean = self.mean.as_ref().expect("StandardScaler must be fitted before transform");
            let scale = self.scale.as_ref().expect("StandardScaler must be fitted before transform");
            let (nrows, ncols) = x.dim();
            let mut out = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    out[[r, c]] = (x[[r, c]] - mean[c]) / scale[c];
                }
            }
            out
        }

        pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
            self.fit(x);
            self.transform(x)
        }
    }

    /// MinMaxScaler scales features to a given range (typically 0 to 1).
    pub struct MinMaxScaler {
        pub min: Option<Array1<f64>>,
        pub max: Option<Array1<f64>>,
    }

    impl MinMaxScaler {
        pub fn new() -> Self {
            Self { min: None, max: None }
        }

        pub fn fit(&mut self, x: &Array2<f64>) {
            let (nrows, ncols) = x.dim();
            if nrows == 0 {
                return;
            }
            let mut mins = Array1::from_elem(ncols, f64::INFINITY);
            let mut maxs = Array1::from_elem(ncols, f64::NEG_INFINITY);
            for r in 0..nrows {
                for c in 0..ncols {
                    let v = x[[r, c]];
                    if v < mins[c] {
                        mins[c] = v;
                    }
                    if v > maxs[c] {
                        maxs[c] = v;
                    }
                }
            }
            self.min = Some(mins);
            self.max = Some(maxs);
        }

        pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
            let min = self.min.as_ref().expect("MinMaxScaler not fitted");
            let max = self.max.as_ref().expect("MinMaxScaler not fitted");
            let (nrows, ncols) = x.dim();
            let mut out = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    let range = max[c] - min[c];
                    let scale = if range > 1e-14 { range } else { 1.0 };
                    out[[r, c]] = (x[[r, c]] - min[c]) / scale;
                }
            }
            out
        }

        pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
            self.fit(x);
            self.transform(x)
        }
    }
}

// --- linear_model submodule ---

pub mod linear_model {
    use super::*;

    /// Ordinary least squares Linear Regression.
    pub struct LinearRegression {
        pub coef: Option<Array1<f64>>,
        pub intercept: Option<f64>,
        pub fit_intercept: bool,
    }

    impl LinearRegression {
        pub fn new(fit_intercept: bool) -> Self {
            Self {
                coef: None,
                intercept: None,
                fit_intercept,
            }
        }

        pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
            let (nrows, ncols) = x.dim();
            if nrows == 0 || nrows != y.len() {
                return Err("Dimensions of X and y must match".to_string());
            }

            if self.fit_intercept {
                let mut x_design = Array2::ones((nrows, ncols + 1));
                for r in 0..nrows {
                    for c in 0..ncols {
                        x_design[[r, c]] = x[[r, c]];
                    }
                }
                let xt = x_design.t().to_owned();
                let xtx = xt.dot(&x_design);
                let xty = xt.dot(y);
                let beta = crate::sp::linalg::solve_vec(&xtx, &xty)?;

                self.coef = Some(Array1::from_vec(beta.slice(ndarray::s![0..ncols]).to_vec()));
                self.intercept = Some(beta[ncols]);
            } else {
                let xt = x.t().to_owned();
                let xtx = xt.dot(x);
                let xty = xt.dot(y);
                let beta = crate::sp::linalg::solve_vec(&xtx, &xty)?;

                self.coef = Some(beta);
                self.intercept = Some(0.0);
            }
            Ok(())
        }

        pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
            let coef = self.coef.as_ref().expect("LinearRegression model not fitted");
            let intercept = self.intercept.expect("LinearRegression model not fitted");
            let mut preds = x.dot(coef);
            preds.mapv_inplace(|v| v + intercept);
            preds
        }
    }

    /// Logistic Regression classifier using Gradient Descent.
    pub struct LogisticRegression {
        pub coef: Option<Array1<f64>>,
        pub intercept: Option<f64>,
        pub max_iter: usize,
        pub lr: f64,
        pub tol: f64,
        pub c_reg: f64, // C parameter (inverse regularization strength)
    }

    impl LogisticRegression {
        pub fn new(max_iter: usize, lr: f64, c_reg: f64) -> Self {
            Self {
                coef: None,
                intercept: None,
                max_iter,
                lr,
                tol: 1e-4,
                c_reg,
            }
        }

        fn sigmoid(z: f64) -> f64 {
            1.0 / (1.0 + (-z).exp())
        }

        pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
            let (nrows, ncols) = x.dim();
            let mut w = Array1::zeros(ncols);
            let mut b = 0.0;
            let lambda = 1.0 / self.c_reg;

            for _ in 0..self.max_iter {
                let mut z = x.dot(&w);
                z.mapv_inplace(|v| v + b);
                let p = z.mapv(Self::sigmoid);
                let diff = &p - y;

                let xt = x.t().to_owned();
                let mut dw = xt.dot(&diff) / (nrows as f64);
                if lambda > 0.0 {
                    dw = dw + &w * lambda;
                }
                let db = diff.sum() / (nrows as f64);

                let mut max_grad_change = 0.0;
                for i in 0..ncols {
                    let step = self.lr * dw[i];
                    if step.abs() > max_grad_change {
                        max_grad_change = step.abs();
                    }
                    w[i] -= step;
                }
                let b_step = self.lr * db;
                if b_step.abs() > max_grad_change {
                    max_grad_change = b_step.abs();
                }
                b -= b_step;

                if max_grad_change < self.tol {
                    break;
                }
            }

            self.coef = Some(w);
            self.intercept = Some(b);
        }

        pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
            let w = self.coef.as_ref().expect("LogisticRegression not fitted");
            let b = self.intercept.expect("LogisticRegression not fitted");
            let (nrows, _) = x.dim();
            let mut z = x.dot(w);
            z.mapv_inplace(|v| v + b);
            let p = z.mapv(Self::sigmoid);

            let mut proba = Array2::zeros((nrows, 2));
            for r in 0..nrows {
                proba[[r, 0]] = 1.0 - p[r];
                proba[[r, 1]] = p[r];
            }
            proba
        }

        pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
            let proba = self.predict_proba(x);
            let (nrows, _) = proba.dim();
            let mut preds = Array1::zeros(nrows);
            for r in 0..nrows {
                preds[r] = if proba[[r, 1]] >= 0.5 { 1.0 } else { 0.0 };
            }
            preds
        }
    }
}

// --- cluster submodule ---

pub mod cluster {
    use super::*;

    /// K-Means clustering.
    pub struct KMeans {
        pub n_clusters: usize,
        pub max_iter: usize,
        pub tol: f64,
        pub cluster_centers: Option<Array2<f64>>,
    }

    impl KMeans {
        pub fn new(n_clusters: usize, max_iter: usize) -> Self {
            Self {
                n_clusters,
                max_iter,
                tol: 1e-4,
                cluster_centers: None,
            }
        }

        pub fn fit(&mut self, x: &Array2<f64>) {
            let (nrows, ncols) = x.dim();
            if nrows < self.n_clusters {
                panic!("Number of samples must be >= n_clusters");
            }

            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..nrows).collect();
            indices.shuffle(&mut rng);

            let mut centroids = Array2::zeros((self.n_clusters, ncols));
            for k in 0..self.n_clusters {
                let idx = indices[k];
                for c in 0..ncols {
                    centroids[[k, c]] = x[[idx, c]];
                }
            }

            let mut labels = vec![0; nrows];

            for _ in 0..self.max_iter {
                let mut changed = false;
                for r in 0..nrows {
                    let mut min_dist = f64::INFINITY;
                    let mut best_k = 0;
                    for k in 0..self.n_clusters {
                        let mut dist_sq = 0.0;
                        for c in 0..ncols {
                            dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                        }
                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                            best_k = k;
                        }
                    }
                    if labels[r] != best_k {
                        labels[r] = best_k;
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }

                let mut new_centroids: Array2<f64> = Array2::zeros((self.n_clusters, ncols));
                let mut counts = vec![0.0; self.n_clusters];
                for r in 0..nrows {
                    let k = labels[r];
                    counts[k] += 1.0;
                    for c in 0..ncols {
                        new_centroids[[k, c]] += x[[r, c]];
                    }
                }

                let mut centroid_diff: f64 = 0.0;
                for k in 0..self.n_clusters {
                    if counts[k] > 0.0 {
                        for c in 0..ncols {
                            let new_val = new_centroids[[k, c]] / counts[k];
                            let diff = centroids[[k, c]] - new_val;
                            centroid_diff += diff * diff;
                            centroids[[k, c]] = new_val;
                        }
                    }
                }

                if centroid_diff.sqrt() < self.tol {
                    break;
                }
            }

            self.cluster_centers = Some(centroids);
        }

        pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
            let centroids = self.cluster_centers.as_ref().expect("KMeans model not fitted");
            let (nrows, ncols) = x.dim();
            let mut labels = Array1::zeros(nrows);
            for r in 0..nrows {
                let mut min_dist = f64::INFINITY;
                let mut best_k = 0;
                for k in 0..self.n_clusters {
                    let mut dist_sq = 0.0;
                    for c in 0..ncols {
                        dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                    }
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_k = k;
                    }
                }
                labels[r] = best_k;
            }
            labels
        }
    }
}

// --- decomposition submodule ---

pub mod decomposition {
    use super::*;

    /// Principal Component Analysis (PCA).
    pub struct PCA {
        pub n_components: usize,
        pub components: Option<Array2<f64>>,
        pub mean: Option<Array1<f64>>,
        pub explained_variance: Option<Array1<f64>>,
    }

    impl PCA {
        pub fn new(n_components: usize) -> Self {
            Self {
                n_components,
                components: None,
                mean: None,
                explained_variance: None,
            }
        }

        pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), String> {
            let (nrows, ncols) = x.dim();
            if nrows <= 1 {
                return Err("PCA requires at least 2 samples".to_string());
            }

            let mean = x.mean_axis(Axis(0)).unwrap();
            let mut x_centered = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    x_centered[[r, c]] = x[[r, c]] - mean[c];
                }
            }

            let svd_res = crate::sp::linalg::svd(&x_centered);
            let vh = svd_res.vh.ok_or_else(|| "SVD failed to compute V^T".to_string())?;

            let k = std::cmp::min(self.n_components, ncols);
            let mut components = Array2::zeros((k, ncols));
            for r in 0..k {
                for c in 0..ncols {
                    components[[r, c]] = vh[[r, c]];
                }
            }

            let s = svd_res.s;
            let mut explained_variance = Array1::zeros(k);
            for i in 0..k {
                explained_variance[i] = s[i].powi(2) / ((nrows - 1) as f64);
            }

            self.mean = Some(mean);
            self.components = Some(components);
            self.explained_variance = Some(explained_variance);
            Ok(())
        }

        pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
            let mean = self.mean.as_ref().expect("PCA model not fitted");
            let components = self.components.as_ref().expect("PCA model not fitted");
            let (nrows, ncols) = x.dim();
            let mut x_centered = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    x_centered[[r, c]] = x[[r, c]] - mean[c];
                }
            }
            x_centered.dot(&components.t())
        }

        pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
            self.fit(x)?;
            Ok(self.transform(x))
        }
    }
}

// --- metrics submodule ---

pub mod metrics {
    use super::*;

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
}

// --- model_selection submodule ---

pub mod model_selection {
    use super::*;

    /// Split arrays or matrices into random train and test subsets.
    pub fn train_test_split<T: Clone + Default>(
        x: &Array2<T>,
        y: &Array1<T>,
        test_size: f64,
        shuffle: bool,
    ) -> (Array2<T>, Array2<T>, Array1<T>, Array1<T>) {
        let n = y.len();
        assert_eq!(x.nrows(), n, "X and y must have same number of samples");
        let mut indices: Vec<usize> = (0..n).collect();
        if shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        let n_test = (n as f64 * test_size).round() as usize;
        let n_train = n - n_test;

        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..n];

        let ncols = x.ncols();

        let mut x_train = Array2::from_elem((n_train, ncols), T::default());
        let mut y_train = Array1::from_elem(n_train, T::default());
        for (i, &idx) in train_indices.iter().enumerate() {
            y_train[i] = y[idx].clone();
            for c in 0..ncols {
                x_train[[i, c]] = x[[idx, c]].clone();
            }
        }

        let mut x_test = Array2::from_elem((n_test, ncols), T::default());
        let mut y_test = Array1::from_elem(n_test, T::default());
        for (i, &idx) in test_indices.iter().enumerate() {
            y_test[i] = y[idx].clone();
            for c in 0..ncols {
                x_test[[i, c]] = x[[idx, c]].clone();
            }
        }

        (x_train, x_test, y_train, y_test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standard_scaler() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = preprocessing::StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);

        let mean = x_scaled.mean_axis(Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-9);
        assert!(mean[1].abs() < 1e-9);
    }

    #[test]
    fn test_linear_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut reg = linear_model::LinearRegression::new(true);
        reg.fit(&x, &y).unwrap();

        let coef = reg.coef.as_ref().unwrap();
        let intercept = reg.intercept.unwrap();

        assert!((coef[0] - 2.0).abs() < 1e-9);
        assert!((intercept - 1.0).abs() < 1e-9);

        let preds = reg.predict(&array![[5.0]]);
        assert!((preds[0] - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_logistic_regression() {
        let x = array![[0.1], [0.2], [0.8], [0.9]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mut clf = linear_model::LogisticRegression::new(1000, 0.5, 1.0);
        clf.fit(&x, &y);

        let preds = clf.predict(&array![[0.15], [0.85]]);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_kmeans() {
        let x = array![[1.0, 1.0], [1.5, 1.5], [10.0, 10.0], [10.5, 10.5]];
        let mut km = cluster::KMeans::new(2, 100);
        km.fit(&x);
        let preds = km.predict(&x);
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[2], preds[3]);
        assert_ne!(preds[0], preds[2]);
    }

    #[test]
    fn test_pca() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let mut pca = decomposition::PCA::new(1);
        pca.fit(&x).unwrap();
        let transformed = pca.transform(&x);
        assert_eq!(transformed.dim(), (3, 1));
    }

    #[test]
    fn test_train_test_split() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (x_train, x_test, y_train, y_test) = model_selection::train_test_split(&x, &y, 0.4, false);
        assert_eq!(x_train.nrows(), 3);
        assert_eq!(x_test.nrows(), 2);
        assert_eq!(y_train.len(), 3);
        assert_eq!(y_test.len(), 2);
    }
}


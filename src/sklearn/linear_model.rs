use ndarray::{Array1, Array2};

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

/// Ridge Regression (L2-regularized Ordinary Least Squares).
pub struct Ridge {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub coef: Option<Array1<f64>>,
    pub intercept: Option<f64>,
}

impl Ridge {
    pub fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            coef: None,
            intercept: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let (nrows, ncols) = x.dim();
        if nrows == 0 || nrows != y.len() {
            return Err("Dimensions of X and y must match".to_string());
        }

        if self.fit_intercept {
            let x_mean = x.mean_axis(ndarray::Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);

            let mut x_centered = Array2::zeros((nrows, ncols));
            for r in 0..nrows {
                for c in 0..ncols {
                    x_centered[[r, c]] = x[[r, c]] - x_mean[c];
                }
            }
            let y_centered = y - y_mean;

            let xt = x_centered.t().to_owned();
            let xtx = xt.dot(&x_centered);
            let xty = xt.dot(&y_centered);

            let mut reg_matrix: Array2<f64> = Array2::eye(ncols);
            reg_matrix.mapv_inplace(|v| v * self.alpha);
            let xtx_reg = xtx + reg_matrix;

            let w = crate::sp::linalg::solve_vec(&xtx_reg, &xty)?;

            let mut dot_product = 0.0;
            for i in 0..ncols {
                dot_product += w[i] * x_mean[i];
            }

            self.coef = Some(w);
            self.intercept = Some(y_mean - dot_product);
        } else {
            let xt = x.t().to_owned();
            let xtx = xt.dot(x);
            let xty = xt.dot(y);

            let mut reg_matrix: Array2<f64> = Array2::eye(ncols);
            reg_matrix.mapv_inplace(|v| v * self.alpha);
            let xtx_reg = xtx + reg_matrix;

            let w = crate::sp::linalg::solve_vec(&xtx_reg, &xty)?;

            self.coef = Some(w);
            self.intercept = Some(0.0);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let coef = self.coef.as_ref().expect("Ridge model not fitted");
        let intercept = self.intercept.expect("Ridge model not fitted");
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
    pub c_reg: f64,
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

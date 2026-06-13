use ndarray::{Array1, Array2, Axis};

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

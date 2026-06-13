use ndarray::{Array1, Array2, Axis};

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

/// OneHotEncoder encodes categorical features as a one-hot numeric array.
pub struct OneHotEncoder {
    pub categories: Vec<Vec<f64>>,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder.
    pub fn new() -> Self {
        Self { categories: Vec::new() }
    }

    /// Fit OneHotEncoder to X.
    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        self.categories = Vec::with_capacity(ncols);
        for col in 0..ncols {
            let mut unique_vals: Vec<f64> = (0..nrows).map(|r| x[[r, col]]).collect();
            unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_vals.dedup();
            self.categories.push(unique_vals);
        }
    }

    /// Transform X using one-hot encoding.
    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let (nrows, ncols) = x.dim();
        let total_out_cols: usize = self.categories.iter().map(|c| c.len()).sum();
        let mut out = Array2::zeros((nrows, total_out_cols));

        for r in 0..nrows {
            let mut current_col = 0;
            for col in 0..ncols {
                let val = x[[r, col]];
                let cat_list = &self.categories[col];
                let cat_idx = cat_list.iter().position(|&x| x == val).unwrap_or(0);
                out[[r, current_col + cat_idx]] = 1.0;
                current_col += cat_list.len();
            }
        }
        out
    }

    /// Fit to X, then transform it.
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x)
    }
}


use ndarray::Array1;

/// 1D linear interpolation.
pub struct Interp1D {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Interp1D {
    /// Create a new 1D linear interpolator from given x and y values.
    pub fn new(x: &Array1<f64>, y: &Array1<f64>) -> Result<Self, String> {
        let n = x.len();
        if n != y.len() {
            return Err("x and y must have the same length".to_string());
        }
        if n < 2 {
            return Err("At least 2 points are required for interpolation".to_string());
        }

        let mut pairs: Vec<(f64, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut x_sorted = Vec::with_capacity(n);
        let mut y_sorted = Vec::with_capacity(n);
        for (xp, yp) in pairs {
            x_sorted.push(xp);
            y_sorted.push(yp);
        }

        Ok(Self { x: x_sorted, y: y_sorted })
    }

    /// Evaluate the interpolator at given new x values.
    pub fn call(&self, x_new: &Array1<f64>) -> Array1<f64> {
        let n = x_new.len();
        let mut y_new = Array1::zeros(n);
        for i in 0..n {
            y_new[i] = self.eval(x_new[i]);
        }
        y_new
    }

    /// Evaluate the interpolator at a single x value.
    pub fn eval(&self, xp: f64) -> f64 {
        let n = self.x.len();
        if xp <= self.x[0] {
            return self.y[0];
        }
        if xp >= self.x[n - 1] {
            return self.y[n - 1];
        }

        let idx = match self.x.binary_search_by(|val| val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        let x0 = self.x[idx - 1];
        let x1 = self.x[idx];
        let y0 = self.y[idx - 1];
        let y1 = self.y[idx];

        y0 + (xp - x0) * (y1 - y0) / (x1 - x0)
    }
}

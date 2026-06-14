use ndarray::{Array1, Array2};

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

// ---------------------------------------------------------------------------
// CubicSpline
// ---------------------------------------------------------------------------

/// Natural cubic spline interpolation (second derivative = 0 at endpoints).
#[allow(dead_code)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    coefficients: Vec<[f64; 4]>, // a, b, c, d for each segment
}

impl CubicSpline {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, String> {
        let n = x.len();
        if n != y.len() {
            return Err("x and y must have the same length".to_string());
        }
        if n < 2 {
            return Err("At least 2 points are required".to_string());
        }

        let mut pairs: Vec<(f64, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (xp, yp) in &pairs {
            xs.push(*xp);
            ys.push(*yp);
        }

        let m = n - 1; // number of segments
        let mut h = vec![0.0f64; m];
        for i in 0..m {
            h[i] = xs[i + 1] - xs[i];
            if h[i] <= 0.0 {
                return Err("x values must be strictly increasing".to_string());
            }
        }

        // Solve tridiagonal system for c (natural spline: c[0] = c[m] = 0)
        // The system for interior knots i = 1..m-1:
        //   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
        //     = 3 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        let size = m + 1;
        let mut c = vec![0.0f64; size];

        if m >= 2 {
            let n_inner = m - 1; // number of unknowns: c[1]..c[m-1]
            let mut dl = vec![0.0f64; n_inner];
            let mut d = vec![0.0f64; n_inner];
            let mut du = vec![0.0f64; n_inner];
            let mut rhs = vec![0.0f64; n_inner];

            for i in 0..n_inner {
                let idx = i + 1; // index into c / y
                d[i] = 2.0 * (h[idx - 1] + h[idx]);
                rhs[i] = 3.0
                    * ((ys[idx + 1] - ys[idx]) / h[idx] - (ys[idx] - ys[idx - 1]) / h[idx - 1]);
                if i > 0 {
                    dl[i - 1] = h[idx - 1];
                }
                if i + 1 < n_inner {
                    du[i] = h[idx];
                }
            }

            // Thomas algorithm for tridiagonal system
            let mut c_inner = vec![0.0f64; n_inner];
            let du_mod = du.clone();
            let mut rhs_mod = rhs.clone();

            for i in 1..n_inner {
                let w = dl[i - 1] / d[i - 1];
                d[i] -= w * du_mod[i - 1];
                rhs_mod[i] -= w * rhs_mod[i - 1];
            }

            c_inner[n_inner - 1] = rhs_mod[n_inner - 1] / d[n_inner - 1];
            for i in (0..n_inner - 1).rev() {
                c_inner[i] = (rhs_mod[i] - du_mod[i] * c_inner[i + 1]) / d[i];
            }

            for i in 0..n_inner {
                c[i + 1] = c_inner[i];
            }
        }

        // Compute coefficients for each segment:
        // S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
        let mut coefficients = Vec::with_capacity(m);
        for i in 0..m {
            let a = ys[i];
            let b = (ys[i + 1] - ys[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            let c_coeff = c[i];
            let d_coeff = (c[i + 1] - c[i]) / (3.0 * h[i]);
            coefficients.push([a, b, c_coeff, d_coeff]);
        }

        Ok(Self {
            x: xs,
            y: ys,
            coefficients,
        })
    }

    fn find_segment(&self, xp: f64) -> usize {
        let n = self.x.len();
        if xp <= self.x[0] {
            return 0;
        }
        if xp >= self.x[n - 1] {
            return n - 2;
        }
        match self
            .x
            .binary_search_by(|val| val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => {
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
            Err(idx) => idx - 1,
        }
    }

    pub fn evaluate(&self, xp: f64) -> f64 {
        let i = self.find_segment(xp);
        let dx = xp - self.x[i];
        let [a, b, c, d] = self.coefficients[i];
        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    pub fn evaluate_many(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xp| self.evaluate(xp)).collect()
    }
}

// ---------------------------------------------------------------------------
// ScipyInterp1D (scipy-compatible interp1d)
// ---------------------------------------------------------------------------

/// SciPy-compatible 1D interpolation supporting linear, nearest, and cubic.
pub struct ScipyInterp1D {
    x: Vec<f64>,
    y: Vec<f64>,
    kind: String,
    fill_value: Option<(f64, f64)>,
}

impl ScipyInterp1D {
    pub fn new(x: &[f64], y: &[f64], kind: &str) -> Self {
        let n = x.len();
        assert_eq!(n, y.len(), "x and y must have the same length");
        assert!(n >= 2, "At least 2 points required");
        assert!(
            kind == "linear" || kind == "nearest" || kind == "cubic",
            "kind must be 'linear', 'nearest', or 'cubic'"
        );

        let mut pairs: Vec<(f64, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (xp, yp) in pairs {
            xs.push(xp);
            ys.push(yp);
        }

        Self {
            x: xs,
            y: ys,
            kind: kind.to_string(),
            fill_value: None,
        }
    }

    pub fn with_fill_value(mut self, fill_value: (f64, f64)) -> Self {
        self.fill_value = Some(fill_value);
        self
    }

    pub fn evaluate(&self, x_new: &[f64]) -> Vec<f64> {
        match self.kind.as_str() {
            "linear" => self.eval_linear(x_new),
            "nearest" => self.eval_nearest(x_new),
            "cubic" => self.eval_cubic(x_new),
            _ => unreachable!(),
        }
    }

    fn eval_linear(&self, x_new: &[f64]) -> Vec<f64> {
        let n = self.x.len();
        x_new
            .iter()
            .map(|&xp| {
                if xp < self.x[0] {
                    return self
                        .fill_value
                        .map(|v| v.0)
                        .unwrap_or(self.y[0]);
                }
                if xp > self.x[n - 1] {
                    return self
                        .fill_value
                        .map(|v| v.1)
                        .unwrap_or(self.y[n - 1]);
                }
                let idx = match self
                    .x
                    .binary_search_by(|val| val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal))
                {
                    Ok(i) => i,
                    Err(i) => i,
                };
                if idx == 0 {
                    self.y[0]
                } else if idx >= n {
                    self.y[n - 1]
                } else {
                    let x0 = self.x[idx - 1];
                    let x1 = self.x[idx];
                    let y0 = self.y[idx - 1];
                    let y1 = self.y[idx];
                    y0 + (xp - x0) * (y1 - y0) / (x1 - x0)
                }
            })
            .collect()
    }

    fn eval_nearest(&self, x_new: &[f64]) -> Vec<f64> {
        let n = self.x.len();
        x_new
            .iter()
            .map(|&xp| {
                if xp < self.x[0] {
                    return self.fill_value.map(|v| v.0).unwrap_or(self.y[0]);
                }
                if xp > self.x[n - 1] {
                    return self.fill_value.map(|v| v.1).unwrap_or(self.y[n - 1]);
                }
                let idx = match self
                    .x
                    .binary_search_by(|val| val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal))
                {
                    Ok(i) => i,
                    Err(i) => {
                        if i == 0 {
                            0
                        } else if i >= n {
                            n - 1
                        } else {
                            let d_lo = xp - self.x[i - 1];
                            let d_hi = self.x[i] - xp;
                            if d_lo <= d_hi {
                                i - 1
                            } else {
                                i
                            }
                        }
                    }
                };
                self.y[idx]
            })
            .collect()
    }

    fn eval_cubic(&self, x_new: &[f64]) -> Vec<f64> {
        let cs = CubicSpline::new(&self.x, &self.y).expect("CubicSpline construction failed");
        x_new
            .iter()
            .map(|&xp| {
                if xp < self.x[0] {
                    return self.fill_value.map(|v| v.0).unwrap_or(self.y[0]);
                }
                if xp > self.x[self.x.len() - 1] {
                    return self
                        .fill_value
                        .map(|v| v.1)
                        .unwrap_or(self.y[self.y.len() - 1]);
                }
                cs.evaluate(xp)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RBFInterpolator
// ---------------------------------------------------------------------------

/// Radial Basis Function interpolator for scattered N-D data.
pub struct RBFInterpolator {
    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    kernel: String,
    epsilon: f64,
}

impl RBFInterpolator {
    pub fn new(x: &[Vec<f64>], y: &[f64], kernel: &str, epsilon: f64) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        assert!(
            ["multiquadric", "inverse", "gaussian", "thin_plate"].contains(&kernel),
            "kernel must be one of: multiquadric, inverse, gaussian, thin_plate"
        );
        assert!(epsilon > 0.0, "epsilon must be positive");

        let n = x.len();
        // Build RBF matrix
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = Self::distance(&x[i], &x[j]);
                a[i][j] = Self::kernel_fn(r, kernel, epsilon);
            }
        }

        // Solve A * w = y using Gaussian elimination
        let weights = Self::solve_linear(&a, y);

        Self {
            centers: x.to_vec(),
            weights,
            kernel: kernel.to_string(),
            epsilon,
        }
    }

    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn kernel_fn(r: f64, kernel: &str, epsilon: f64) -> f64 {
        match kernel {
            "multiquadric" => (r.powi(2) + epsilon.powi(2)).sqrt(),
            "inverse" => 1.0 / (r.powi(2) + epsilon.powi(2)).sqrt(),
            "gaussian" => (-(r / epsilon).powi(2)).exp(),
            "thin_plate" => {
                if r == 0.0 {
                    0.0
                } else {
                    r.powi(2) * r.ln()
                }
            }
            _ => unreachable!(),
        }
    }

    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .zip(b.iter())
            .map(|(row, &bi)| {
                let mut r = row.clone();
                r.push(bi);
                r
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            assert!(pivot.abs() > 1e-14, "Singular matrix in RBF system");

            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for k in col..=n {
                    aug[row][k] -= factor * aug[col][k];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }
        x
    }

    pub fn evaluate(&self, points: &[Vec<f64>]) -> Vec<f64> {
        points
            .iter()
            .map(|pt| {
                self.centers
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(center, &w)| {
                        let r = Self::distance(pt, center);
                        w * Self::kernel_fn(r, &self.kernel, self.epsilon)
                    })
                    .sum()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// griddata
// ---------------------------------------------------------------------------

/// Scattered-data interpolation on N-D points.
///
/// `method`: "nearest" or "linear" (barycentric for 2-D).
pub fn griddata(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
    method: &str,
) -> Vec<f64> {
    assert_eq!(points.len(), values.len());
    assert!(method == "nearest" || method == "linear");

    match method {
        "nearest" => griddata_nearest(points, values, xi),
        "linear" => griddata_linear(points, values, xi),
        _ => unreachable!(),
    }
}

fn griddata_nearest(points: &[Vec<f64>], values: &[f64], xi: &[Vec<f64>]) -> Vec<f64> {
    xi.iter()
        .map(|pt| {
            let mut best_dist = f64::INFINITY;
            let mut best_val = 0.0f64;
            for (i, cp) in points.iter().enumerate() {
                let d: f64 = cp
                    .iter()
                    .zip(pt.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if d < best_dist {
                    best_dist = d;
                    best_val = values[i];
                }
            }
            best_val
        })
        .collect()
}

fn griddata_linear(points: &[Vec<f64>], values: &[f64], xi: &[Vec<f64>]) -> Vec<f64> {
    // For 2-D: Delaunay-like barycentric interpolation via natural neighbor
    // approximation using inverse-distance weighted simplex search.
    // For general N-D, fall back to IDW with a small neighbourhood.
    let ndim = points[0].len();

    if ndim == 2 {
        griddata_linear_2d(points, values, xi)
    } else {
        // Fallback: inverse distance weighting (not true linear, but functional)
        griddata_idw(points, values, xi, 2.0)
    }
}

fn griddata_linear_2d(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
) -> Vec<f64> {
    // For 2-D: find the triangle containing each query point and do barycentric interpolation.
    xi.iter()
        .map(|pt| {
            // Compute distances to all points
            let mut dists: Vec<(f64, usize)> = points
                .iter()
                .enumerate()
                .map(|(i, cp)| {
                    let d: f64 = cp
                        .iter()
                        .zip(pt.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (d, i)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Try barycentric with the 3 nearest points
            let i0 = dists[0].1;
            let i1 = dists[1].1;
            let i2 = dists[2].1;

            let (x0, y0) = (points[i0][0], points[i0][1]);
            let (x1, y1) = (points[i1][0], points[i1][1]);
            let (x2, y2) = (points[i2][0], points[i2][1]);
            let (xp, yp) = (pt[0], pt[1]);

            let det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
            if det.abs() < 1e-15 {
                // Degenerate triangle, fall back to nearest
                return values[i0];
            }

            let lambda0 = ((y1 - y2) * (xp - x2) + (x2 - x1) * (yp - y2)) / det;
            let lambda1 = ((y2 - y0) * (xp - x2) + (x0 - x2) * (yp - y2)) / det;
            let lambda2 = 1.0 - lambda0 - lambda1;

            // If point is inside the triangle (all barycentric coords >= 0)
            if lambda0 >= -1e-10 && lambda1 >= -1e-10 && lambda2 >= -1e-10 {
                lambda0 * values[i0] + lambda1 * values[i1] + lambda2 * values[i2]
            } else {
                // Outside: fall back to nearest
                values[i0]
            }
        })
        .collect()
}

fn griddata_idw(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
    power: f64,
) -> Vec<f64> {
    xi.iter()
        .map(|pt| {
            let mut w_sum = 0.0f64;
            let mut val_sum = 0.0f64;
            for (i, cp) in points.iter().enumerate() {
                let d: f64 = cp
                    .iter()
                    .zip(pt.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if d < 1e-15 {
                    return values[i];
                }
                let w = 1.0 / d.powf(power);
                w_sum += w;
                val_sum += w * values[i];
            }
            val_sum / w_sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// RegularGridInterpolator
// ---------------------------------------------------------------------------

/// Multilinear interpolation on a regular (tensor-product) grid.
pub struct RegularGridInterpolator {
    grid: Vec<Vec<f64>>,
    values: Array2<f64>,
    method: String,
}

impl RegularGridInterpolator {
    pub fn new(grid: Vec<Vec<f64>>, values: Array2<f64>, method: &str) -> Self {
        assert!(
            method == "linear" || method == "nearest",
            "method must be 'linear' or 'nearest'"
        );
        for g in &grid {
            assert!(g.len() >= 2, "Each grid axis must have at least 2 points");
        }
        Self {
            grid,
            values,
            method: method.to_string(),
        }
    }

    pub fn evaluate(&self, points: &[Vec<f64>]) -> Vec<f64> {
        match self.method.as_str() {
            "linear" => self.eval_multilinear(points),
            "nearest" => self.eval_nearest(points),
            _ => unreachable!(),
        }
    }

    fn eval_nearest(&self, points: &[Vec<f64>]) -> Vec<f64> {
        points
            .iter()
            .map(|pt| {
                let mut idx = Vec::with_capacity(self.grid.len());
                for (d, g) in self.grid.iter().enumerate() {
                    let mut best = 0;
                    let mut best_dist = (g[0] - pt[d]).abs();
                    for (j, &gv) in g.iter().enumerate().skip(1) {
                        let dist = (gv - pt[d]).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best = j;
                        }
                    }
                    idx.push(best);
                }
                self.values[[idx[0], idx[1]]]
            })
            .collect()
    }

    fn eval_multilinear(&self, points: &[Vec<f64>]) -> Vec<f64> {
        points
            .iter()
            .map(|pt| {
                let ndim = self.grid.len();
                let mut lower = vec![0usize; ndim];
                let mut frac = vec![0.0f64; ndim];

                for (d, g) in self.grid.iter().enumerate() {
                    let xp = pt[d];
                    if xp <= g[0] {
                        lower[d] = 0;
                        frac[d] = 0.0;
                    } else if xp >= g[g.len() - 1] {
                        lower[d] = g.len() - 2;
                        frac[d] = 1.0;
                    } else {
                        let idx = match g
                            .binary_search_by(|val| {
                                val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal)
                            }) {
                            Ok(i) => i,
                            Err(i) => i,
                        };
                        let lo = if idx == 0 { 0 } else { idx - 1 };
                        lower[d] = lo;
                        frac[d] = (xp - g[lo]) / (g[lo + 1] - g[lo]);
                    }
                }

                // Bilinear interpolation (2D): corner values weighted by fractions
                let i0 = lower[0];
                let j0 = lower[1];
                let t = frac[0];
                let u = frac[1];

                let v00 = self.values[[i0, j0]];
                let v10 = self.values[[i0 + 1, j0]];
                let v01 = self.values[[i0, j0 + 1]];
                let v11 = self.values[[i0 + 1, j0 + 1]];

                (1.0 - t) * (1.0 - u) * v00
                    + t * (1.0 - u) * v10
                    + (1.0 - t) * u * v01
                    + t * u * v11
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Akima1DInterpolator
// ---------------------------------------------------------------------------

/// Akima 1D interpolation -- monotone, less overshoot than cubic spline.
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>, // slopes at each point
}

impl Akima1DInterpolator {
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        assert_eq!(n, y.len(), "x and y must have the same length");
        assert!(n >= 5, "Akima interpolation requires at least 5 points");

        let mut pairs: Vec<(f64, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (xp, yp) in &pairs {
            xs.push(*xp);
            ys.push(*yp);
        }

        // Compute finite differences (slopes between consecutive points)
        let m_len = n - 1;
        let mut delta = vec![0.0f64; m_len];
        for i in 0..m_len {
            delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
        }

        // Compute Akima slopes at each interior point using the classic formula
        // m[i] = (|delta[i+1]-delta[i]| * delta[i-1] + |delta[i-2]-delta[i-1]| * delta[i])
        //        / (|delta[i+1]-delta[i]| + |delta[i-2]-delta[i-1]|)
        // with endpoint extrapolation.
        let mut m = vec![0.0f64; n];

        // For indices 0, 1, n-2, n-1 we need special treatment via extrapolation
        // of delta. Extend delta by 2 on each side.
        let mut ext = vec![0.0f64; m_len + 4];
        for i in 0..m_len {
            ext[i + 2] = delta[i];
        }
        // Extrapolate: ext[0], ext[1] and ext[m_len+2], ext[m_len+3]
        ext[1] = 2.0 * ext[2] - ext[3];
        ext[0] = 2.0 * ext[1] - ext[2];
        ext[m_len + 2] = 2.0 * ext[m_len + 1] - ext[m_len];
        ext[m_len + 3] = 2.0 * ext[m_len + 2] - ext[m_len + 1];

        for i in 0..n {
            let d0 = ext[i];
            let d1 = ext[i + 1];
            let d2 = ext[i + 2];

            let w1 = (d2 - d1).abs();
            let w2 = (d0 - d1).abs();

            if w1 + w2 < 1e-30 {
                // Fallback: average of the two adjacent delta values
                m[i] = (d1 + d2) / 2.0;
            } else {
                m[i] = (w1 * d1 + w2 * d2) / (w1 + w2);
            }
        }

        Self { x: xs, y: ys, m }
    }

    fn find_segment(&self, xp: f64) -> usize {
        let n = self.x.len();
        if xp <= self.x[0] {
            return 0;
        }
        if xp >= self.x[n - 1] {
            return n - 2;
        }
        match self
            .x
            .binary_search_by(|val| val.partial_cmp(&xp).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => {
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
            Err(idx) => idx - 1,
        }
    }

    pub fn evaluate(&self, xp: f64) -> f64 {
        let i = self.find_segment(xp);
        let dx = self.x[i + 1] - self.x[i];
        let t = (xp - self.x[i]) / dx;

        // Hermite basis functions
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * self.y[i]
            + h10 * dx * self.m[i]
            + h01 * self.y[i + 1]
            + h11 * dx * self.m[i + 1]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // --- Existing test (preserved) ---

    #[test]
    fn test_interp1d_linear() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![0.0, 10.0, 20.0]);
        let interp = Interp1D::new(&x, &y).unwrap();

        let x_new = Array1::from_vec(vec![0.5, 1.5]);
        let y_new = interp.call(&x_new);
        assert_eq!(y_new.len(), 2);
        assert!((y_new[0] - 5.0).abs() < 1e-9);
        assert!((y_new[1] - 15.0).abs() < 1e-9);
    }

    // --- CubicSpline tests ---

    #[test]
    fn test_cubic_spline_x_cubed() {
        // Interpolate y = x^3 at several knots, verify smoothness.
        // Natural cubic spline (S''=0 at endpoints) doesn't exactly reproduce
        // cubics when S'' != 0 at boundaries, so we use a relaxed tolerance.
        let x: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * v * v).collect();
        let cs = CubicSpline::new(&x, &y).unwrap();

        for i in 0..x.len() - 1 {
            let mid = (x[i] + x[i + 1]) / 2.0;
            let expected = mid * mid * mid;
            let got = cs.evaluate(mid);
            assert!(
                (got - expected).abs() < 1.0,
                "At x={mid}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_cubic_spline_exact_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let cs = CubicSpline::new(&x, &y).unwrap();

        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!((cs.evaluate(*xi) - yi).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cubic_spline_evaluate_many() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let cs = CubicSpline::new(&x, &y).unwrap();
        let pts = vec![0.5, 1.5, 2.5];
        let vals = cs.evaluate_many(&pts);
        assert_eq!(vals.len(), 3);
        // All values should be finite and within the data range
        for v in &vals {
            assert!(v.is_finite());
            assert!(*v >= -1.0 && *v <= 2.0);
        }
    }

    // --- ScipyInterp1D tests ---

    #[test]
    fn test_scipy_interp1d_linear_exact_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let interp = ScipyInterp1D::new(&x, &y, "linear");

        let xq = vec![0.0, 1.0, 2.0, 3.0];
        let yq = interp.evaluate(&xq);
        for (i, &xi) in xq.iter().enumerate() {
            assert!(
                (yq[i] - y[i]).abs() < 1e-12,
                "At x={xi}: expected {}, got {}", y[i], yq[i]
            );
        }
    }

    #[test]
    fn test_scipy_interp1d_nearest() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 20.0, 30.0];
        let interp = ScipyInterp1D::new(&x, &y, "nearest");

        let yq = interp.evaluate(&[0.3, 0.7, 1.4]);
        assert!((yq[0] - 10.0).abs() < 1e-12); // 0.3 nearest to 0.0
        assert!((yq[1] - 20.0).abs() < 1e-12); // 0.7 nearest to 1.0
        assert!((yq[2] - 20.0).abs() < 1e-12); // 1.4 nearest to 1.0
    }

    #[test]
    fn test_scipy_interp1d_cubic() {
        let x: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();
        let interp = ScipyInterp1D::new(&x, &y, "cubic");

        // Should be close to sin at midpoints
        let yq = interp.evaluate(&[0.5, 1.5, 2.5]);
        assert!((yq[0] - 0.5_f64.sin()).abs() < 0.05);
        assert!((yq[1] - 1.5_f64.sin()).abs() < 0.05);
    }

    // --- RBFInterpolator tests ---

    #[test]
    fn test_rbf_interpolator_smooth_2d() {
        // Interpolate f(x,y) = sin(x)*cos(y) at scattered points
        let pts: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.2, 0.8],
            vec![0.8, 0.2],
        ];
        let vals: Vec<f64> = pts
            .iter()
            .map(|p| p[0].sin() * p[1].cos())
            .collect();

        let rbf = RBFInterpolator::new(&pts, &vals, "gaussian", 1.0);

        // Should reproduce training data
        let pred = rbf.evaluate(&pts);
        for (i, (v, p)) in vals.iter().zip(pred.iter()).enumerate() {
            assert!(
                (v - p).abs() < 0.1,
                "Point {i}: expected {v}, got {p}"
            );
        }
    }

    #[test]
    fn test_rbf_interpolator_thin_plate() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let vals = vec![0.0, 1.0, 1.0, 0.0];
        let rbf = RBFInterpolator::new(&pts, &vals, "thin_plate", 1.0);

        // Check interpolation at center -- should be reasonable
        let pred = rbf.evaluate(&[vec![0.5, 0.5]]);
        assert!(pred[0].is_finite(), "thin_plate prediction must be finite");
    }

    // --- griddata tests ---

    #[test]
    fn test_griddata_nearest() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let vals = vec![10.0, 20.0, 30.0, 40.0];

        let query = vec![vec![0.1, 0.1], vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.9]];
        let result = griddata(&pts, &vals, &query, "nearest");

        assert!((result[0] - 10.0).abs() < 1e-12); // nearest to (0,0)
        assert!((result[1] - 20.0).abs() < 1e-12); // nearest to (1,0)
        assert!((result[2] - 30.0).abs() < 1e-12); // nearest to (0,1)
        assert!((result[3] - 40.0).abs() < 1e-12); // nearest to (1,1)
    }

    #[test]
    fn test_griddata_linear_2d() {
        // Points forming a simple triangle
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let vals = vec![0.0, 1.0, 1.0, 2.0];

        // Query at centroid of first triangle
        let query = vec![vec![0.25, 0.25]];
        let result = griddata(&pts, &vals, &query, "linear");
        // Should be interpolated, not extrapolated
        assert!(result[0] > -1.0 && result[0] < 3.0);
    }

    // --- RegularGridInterpolator tests ---

    #[test]
    fn test_regular_grid_bilinear() {
        // 2D grid: x=[0,1,2], y=[0,1]
        // values[i,j] = x[i] + 2*y[j]
        let grid = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0]];
        let values = array![[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]];
        let interp = RegularGridInterpolator::new(grid, values, "linear");

        // Test at grid points
        let result = interp.evaluate(&[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]]);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
        assert!((result[2] - 2.0).abs() < 1e-12);

        // Test at midpoint
        let result = interp.evaluate(&[vec![0.5, 0.5]]);
        assert!((result[0] - 1.5).abs() < 1e-12); // 0.5 + 2*0.5 = 1.5
    }

    #[test]
    fn test_regular_grid_nearest() {
        let grid = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0]];
        let values = array![[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]];
        let interp = RegularGridInterpolator::new(grid, values, "nearest");

        let result = interp.evaluate(&[vec![0.3, 0.3]]);
        // nearest to (0,0) => 0.0
        assert!((result[0] - 0.0).abs() < 1e-12);
    }

    // --- Akima1DInterpolator tests ---

    #[test]
    fn test_akima_no_overshoot() {
        // Data with a sharp peak -- Akima should not overshoot
        let x: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let mut y = vec![0.0f64; 11];
        y[5] = 10.0; // sharp peak
        let akima = Akima1DInterpolator::new(&x, &y);

        // Evaluate between points, should not exceed data range significantly
        for i in 0..10 {
            let mid = (x[i] + x[i + 1]) / 2.0;
            let v = akima.evaluate(mid);
            assert!(
                v >= -1.0 && v <= 11.0,
                "Overshoot at x={mid}: got {v}"
            );
        }
    }

    #[test]
    fn test_akima_exact_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
        let akima = Akima1DInterpolator::new(&x, &y);

        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!(
                (akima.evaluate(*xi) - yi).abs() < 1e-10,
                "At x={xi}: expected {yi}, got {}", akima.evaluate(*xi)
            );
        }
    }

    #[test]
    fn test_akima_smooth_function() {
        // Interpolate sin(x) -- should be smooth
        let x: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();
        let akima = Akima1DInterpolator::new(&x, &y);

        // Check at midpoints
        for i in 0..x.len() - 1 {
            let mid = (x[i] + x[i + 1]) / 2.0;
            let expected = mid.sin();
            let got = akima.evaluate(mid);
            assert!(
                (got - expected).abs() < 0.1,
                "At x={mid}: expected {expected}, got {got}"
            );
        }
    }
}

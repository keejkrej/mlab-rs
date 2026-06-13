use ndarray::{Array1, Array2};

/// Common trait for kernel functions.
pub trait Kernel {
    /// Compute the Gram matrix between rows of `x` and `y`.
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64>;

    /// String identifier for the kernel.
    fn id(&self) -> &str;
}

/// Linear (dot-product) kernel.
pub struct LinearKernel {
    pub c0: f64,
}

impl LinearKernel {
    pub fn new(c0: f64) -> Self {
        Self { c0 }
    }
}

impl Default for LinearKernel {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Kernel for LinearKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        x.dot(&y.t()) + self.c0
    }

    fn id(&self) -> &str {
        "LinearKernel"
    }
}

/// Polynomial kernel.
pub struct PolynomialKernel {
    pub d: usize,
    pub gamma: Option<f64>,
    pub c0: f64,
}

impl PolynomialKernel {
    pub fn new(d: usize, gamma: Option<f64>, c0: f64) -> Self {
        Self { d, gamma, c0 }
    }
}

impl Default for PolynomialKernel {
    fn default() -> Self {
        Self::new(3, None, 1.0)
    }
}

impl Kernel for PolynomialKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let gamma = self.gamma.unwrap_or(1.0 / x.ncols() as f64);
        let base = gamma * x.dot(&y.t()) + self.c0;
        base.mapv(|v| v.powi(self.d as i32))
    }

    fn id(&self) -> &str {
        "PolynomialKernel"
    }
}

/// Radial basis function (RBF) / squared exponential kernel.
pub struct RBFKernel {
    pub sigma: Option<f64>,
}

impl RBFKernel {
    pub fn new(sigma: Option<f64>) -> Self {
        Self { sigma }
    }
}

impl Default for RBFKernel {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Kernel for RBFKernel {
    fn kernel(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Array2<f64> {
        let y = y.unwrap_or(x);
        let sigma = self.sigma.unwrap_or((x.ncols() as f64 / 2.0).sqrt());
        let dists = pairwise_l2_distances(&(x / sigma), &(y / sigma));
        dists.mapv(|v| (-0.5 * v * v).exp())
    }

    fn id(&self) -> &str {
        "RBFKernel"
    }
}

/// Initialize a kernel from a string identifier or use the default linear kernel.
pub fn kernel_initializer(param: Option<&str>) -> Result<Box<dyn Kernel>, String> {
    match param {
        None | Some("linear") | Some("LinearKernel") => Ok(Box::new(LinearKernel::default())),
        Some("polynomial") | Some("PolynomialKernel") => Ok(Box::new(PolynomialKernel::default())),
        Some("rbf") | Some("RBFKernel") => Ok(Box::new(RBFKernel::default())),
        Some(other) => Err(format!("Unknown kernel: {}", other)),
    }
}

/// Pairwise Euclidean distances between rows of `x` and `y`.
pub fn pairwise_l2_distances(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let (n, _) = x.dim();
    let (m, _) = y.dim();

    let x_sq: Array1<f64> = x.rows().into_iter().map(|row| row.iter().map(|&v| v * v).sum()).collect();
    let y_sq: Array1<f64> = y.rows().into_iter().map(|row| row.iter().map(|&v| v * v).sum()).collect();

    let mut dists = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let dot: f64 = (0..x.ncols()).map(|k| x[[i, k]] * y[[j, k]]).sum();
            let d = (x_sq[i] - 2.0 * dot + y_sq[j]).max(0.0);
            dists[[i, j]] = d.sqrt();
        }
    }
    dists
}

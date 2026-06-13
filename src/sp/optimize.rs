use ndarray::{Array1, Array2};

/// Golden-section search for a 1D scalar minimization problem.
pub fn rsminimize_scalar<F>(f: F, bounds: (f64, f64), tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = bounds;
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / golden_ratio;

    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d);
        }
    }

    (a + b) / 2.0
}

/// Fit model parameters using a simple Gauss-Newton least-squares update.
pub fn curve_fit<F>(f: F, xdata: &Array1<f64>, ydata: &Array1<f64>, p0: &Array1<f64>) -> Result<Array1<f64>, String>
where
    F: Fn(&Array1<f64>, f64) -> f64,
{
    let n = xdata.len();
    let p = p0.len();
    if n == 0 || p == 0 {
        return Ok(p0.clone());
    }
    assert_eq!(n, ydata.len(), "xdata and ydata must have the same length");

    let mut params = p0.to_vec();
    let eps = 1e-6;
    for _ in 0..50 {
        let mut jtj = Array2::<f64>::zeros((p, p));
        let mut jtr = Array1::<f64>::zeros(p);
        for i in 0..n {
            let yhat = f(&Array1::from_vec(params.clone()), xdata[i]);
            let residual = ydata[i] - yhat;
            for j in 0..p {
                let mut plus = params.clone();
                plus[j] += eps;
                let mut minus = params.clone();
                minus[j] -= eps;
                let jac = (f(&Array1::from_vec(plus), xdata[i]) - f(&Array1::from_vec(minus), xdata[i])) / (2.0 * eps);
                jtj[[j, j]] += jac * jac;
                jtr[j] += jac * residual;
            }
        }
        let mut delta = Array1::<f64>::zeros(p);
        for i in 0..p {
            delta[i] = jtr[i] / (jtj[[i, i]] + 1e-12);
        }
        let step_norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        for i in 0..p {
            params[i] += delta[i];
        }
        if step_norm < 1e-8 {
            break;
        }
    }
    Ok(Array1::from_vec(params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_rsminimize_scalar() {
        let x = rsminimize_scalar(|x| (x - 2.0).powi(2), (0.0, 4.0), 1e-6);
        assert!((x - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_curve_fit() {
        let xdata = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let ydata = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let p0 = Array1::from_vec(vec![1.0, 1.0]);
        let fitted = curve_fit(|params, x| params[0] * x + params[1], &xdata, &ydata, &p0).unwrap();
        assert!((fitted[0] - 2.0).abs() < 0.2);
        assert!((fitted[1] - 1.0).abs() < 0.2);
    }
}

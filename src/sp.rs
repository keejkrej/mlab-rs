use ndarray::{Array1, Array2};
pub use rustfft::num_complex::Complex;

// --- Helper functions for conversion between ndarray and nalgebra ---

fn ndarray_to_nalgebra(arr: &Array2<f64>) -> nalgebra::DMatrix<f64> {
    let (nrows, ncols) = arr.dim();
    nalgebra::DMatrix::from_row_iterator(nrows, ncols, arr.iter().cloned())
}

fn nalgebra_to_ndarray(mat: nalgebra::DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = mat.shape();
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in mat.row_iter() {
        flat.extend(row.iter().cloned());
    }
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

// --- linalg submodule ---

pub mod linalg {
    use super::*;

    pub struct SVDResult {
        pub u: Option<Array2<f64>>,
        pub s: Array1<f64>,
        pub vh: Option<Array2<f64>>,
    }

    /// Inverse of a matrix.
    pub fn inv(arr: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mat = ndarray_to_nalgebra(arr);
        let inv_mat = mat.try_inverse().ok_or_else(|| "Matrix is singular".to_string())?;
        Ok(nalgebra_to_ndarray(inv_mat))
    }

    /// Determinant of a matrix.
    pub fn det(arr: &Array2<f64>) -> f64 {
        let mat = ndarray_to_nalgebra(arr);
        mat.determinant()
    }

    /// Solve a linear matrix equation, or system of linear scalar equations.
    pub fn solve(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
        let a_mat = ndarray_to_nalgebra(a);
        let b_mat = ndarray_to_nalgebra(b);
        let decomp = a_mat.lu();
        let x_mat = decomp.solve(&b_mat).ok_or_else(|| "Matrix solver failed".to_string())?;
        Ok(nalgebra_to_ndarray(x_mat))
    }

    /// Solve a linear system with a 1D vector target.
    pub fn solve_vec(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
        let a_mat = ndarray_to_nalgebra(a);
        let b_vec = nalgebra::DVector::from_iterator(b.len(), b.iter().cloned());
        let decomp = a_mat.lu();
        let x_vec = decomp.solve(&b_vec).ok_or_else(|| "Matrix solver failed".to_string())?;
        Ok(Array1::from_vec(x_vec.iter().cloned().collect()))
    }

    /// Cholesky decomposition.
    pub fn cholesky(arr: &Array2<f64>, lower: bool) -> Result<Array2<f64>, String> {
        let mat = ndarray_to_nalgebra(arr);
        let chol = mat.cholesky().ok_or_else(|| "Matrix is not positive-definite".to_string())?;
        let l_mat = chol.unpack();
        if lower {
            Ok(nalgebra_to_ndarray(l_mat))
        } else {
            Ok(nalgebra_to_ndarray(l_mat.transpose()))
        }
    }

    /// Singular Value Decomposition.
    pub fn svd(arr: &Array2<f64>) -> SVDResult {
        let mat = ndarray_to_nalgebra(arr);
        let svd = mat.svd(true, true);
        let u = svd.u.map(|u_mat| nalgebra_to_ndarray(u_mat));
        let s = Array1::from_vec(svd.singular_values.iter().cloned().collect());
        let vh = svd.v_t.map(|vt_mat| nalgebra_to_ndarray(vt_mat));
        SVDResult { u, s, vh }
    }

    /// Compute singular values of a matrix.
    pub fn svdvals(arr: &Array2<f64>) -> Array1<f64> {
        let mat = ndarray_to_nalgebra(arr);
        let svd = mat.svd(false, false);
        Array1::from_vec(svd.singular_values.iter().cloned().collect())
    }
}

// --- fft submodule ---

pub mod fft {
    use super::*;
    use rustfft::FftPlanner;

    /// 1D Forward Fast Fourier Transform.
    pub fn fft(arr: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
        let mut planner = FftPlanner::new();
        let n = arr.len();
        let fft_op = planner.plan_fft_forward(n);
        let mut buffer = arr.to_vec();
        fft_op.process(&mut buffer);
        Array1::from_vec(buffer)
    }

    /// 1D Inverse Fast Fourier Transform.
    pub fn ifft(arr: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
        let mut planner = FftPlanner::new();
        let n = arr.len();
        let fft_op = planner.plan_fft_inverse(n);
        let mut buffer = arr.to_vec();
        fft_op.process(&mut buffer);
        let scale = 1.0 / (n as f64);
        for val in &mut buffer {
            *val = *val * scale;
        }
        Array1::from_vec(buffer)
    }

    /// Helper for running FFT on real vectors.
    pub fn fft_real(arr: &Array1<f64>) -> Array1<Complex<f64>> {
        let complex_arr = arr.mapv(|x| Complex::new(x, 0.0));
        fft(&complex_arr)
    }

    /// 2D Forward Fast Fourier Transform.
    pub fn fft2(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (nrows, ncols) = arr.dim();
        let mut temp = Array2::zeros((nrows, ncols));
        let mut planner = FftPlanner::new();

        let row_fft = planner.plan_fft_forward(ncols);
        for r in 0..nrows {
            let mut row_buf = arr.row(r).to_vec();
            row_fft.process(&mut row_buf);
            for c in 0..ncols {
                temp[[r, c]] = row_buf[c];
            }
        }

        let col_fft = planner.plan_fft_forward(nrows);
        let mut result = Array2::zeros((nrows, ncols));
        for c in 0..ncols {
            let mut col_buf = temp.column(c).to_vec();
            col_fft.process(&mut col_buf);
            for r in 0..nrows {
                result[[r, c]] = col_buf[r];
            }
        }

        result
    }

    /// 2D Inverse Fast Fourier Transform.
    pub fn ifft2(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (nrows, ncols) = arr.dim();
        let mut temp = Array2::zeros((nrows, ncols));
        let mut planner = FftPlanner::new();

        let row_ifft = planner.plan_fft_inverse(ncols);
        for r in 0..nrows {
            let mut row_buf = arr.row(r).to_vec();
            row_ifft.process(&mut row_buf);
            for c in 0..ncols {
                temp[[r, c]] = row_buf[c];
            }
        }

        let col_ifft = planner.plan_fft_inverse(nrows);
        let mut result = Array2::zeros((nrows, ncols));
        for c in 0..ncols {
            let mut col_buf = temp.column(c).to_vec();
            col_ifft.process(&mut col_buf);
            for r in 0..nrows {
                result[[r, c]] = col_buf[r];
            }
        }

        let scale = 1.0 / ((nrows * ncols) as f64);
        result.mapv(|x| x * scale)
    }
}

// --- stats submodule ---

pub mod stats {
    /// Standard error function approximation (Abramowitz and Stegun).
    pub fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp());
        sign * y
    }

    pub struct Norm;

    impl Norm {
        /// Probability density function of normal distribution.
        pub fn pdf(x: f64, loc: f64, scale: f64) -> f64 {
            let variance = scale * scale;
            let exponent = -((x - loc).powi(2)) / (2.0 * variance);
            (1.0 / (scale * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
        }

        /// Cumulative distribution function of normal distribution.
        pub fn cdf(x: f64, loc: f64, scale: f64) -> f64 {
            0.5 * (1.0 + erf((x - loc) / (scale * 2.0_f64.sqrt())))
        }
    }
}

// --- signal submodule ---

pub mod signal {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum ConvolveMode {
        Full,
        Same,
        Valid,
    }

    /// 1D Convolution of two signals.
    pub fn convolve(in1: &Array1<f64>, in2: &Array1<f64>, mode: ConvolveMode) -> Array1<f64> {
        let n1 = in1.len();
        let n2 = in2.len();
        if n1 == 0 || n2 == 0 {
            return Array1::from_vec(vec![]);
        }

        let out_len = n1 + n2 - 1;
        let mut full_result = vec![0.0; out_len];

        for i in 0..n1 {
            for j in 0..n2 {
                full_result[i + j] += in1[i] * in2[j];
            }
        }

        match mode {
            ConvolveMode::Full => Array1::from_vec(full_result),
            ConvolveMode::Same => {
                let start = (n2 - 1) / 2;
                let end = start + n1;
                Array1::from_vec(full_result[start..end].to_vec())
            }
            ConvolveMode::Valid => {
                if n1 >= n2 {
                    let start = n2 - 1;
                    let end = n1;
                    Array1::from_vec(full_result[start..end].to_vec())
                } else {
                    let start = n1 - 1;
                    let end = n2;
                    Array1::from_vec(full_result[start..end].to_vec())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linalg() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let a_inv = linalg::inv(&a).unwrap();
        let eye = a.dot(&a_inv);
        assert!((eye[[0, 0]] - 1.0).abs() < 1e-9);
        assert!(eye[[0, 1]].abs() < 1e-9);
        assert!(eye[[1, 0]].abs() < 1e-9);
        assert!((eye[[1, 1]] - 1.0).abs() < 1e-9);

        assert!((linalg::det(&a) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_fft() {
        let sig = Array1::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ]);
        let sig_fft = fft::fft(&sig);
        let sig_ifft = fft::ifft(&sig_fft);

        for i in 0..4 {
            assert!((sig[i].re - sig_ifft[i].re).abs() < 1e-9);
            assert!(sig_ifft[i].im.abs() < 1e-9);
        }
    }

    #[test]
    fn test_stats() {
        let p = stats::Norm::cdf(0.0, 0.0, 1.0);
        assert!((p - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_signal() {
        let in1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let in2 = Array1::from_vec(vec![0.0, 1.0, 0.5]);
        let res = signal::convolve(&in1, &in2, signal::ConvolveMode::Full);
        assert_eq!(res.len(), 5);
        assert_eq!(res[0], 0.0);
        assert_eq!(res[1], 1.0);
        assert_eq!(res[2], 2.5);
        assert_eq!(res[3], 4.0);
        assert_eq!(res[4], 1.5);
    }
}


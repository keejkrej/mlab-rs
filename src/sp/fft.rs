use ndarray::{Array1, Array2};
pub use rustfft::num_complex::Complex;
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

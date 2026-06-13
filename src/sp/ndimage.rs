use ndarray::Array1;

/// 1D uniform (box) filter / moving average.
pub fn rsuniform_filter1d(input: &Array1<f64>, size: usize) -> Array1<f64> {
    assert!(size > 0, "size must be positive");
    let n = input.len();
    let mut out = Array1::zeros(n);
    for i in 0..n {
        let start = i;
        let end = (i + size).min(n);
        let window = &input.slice(ndarray::s![start..end]);
        out[i] = window.sum() / (end - start) as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsuniform_filter1d() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = rsuniform_filter1d(&x, 2);
        assert!((y[1] - 2.5).abs() < 1e-9);
    }
}

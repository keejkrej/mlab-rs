pub mod linalg;
pub mod fft;
pub mod stats;
pub mod signal;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use fft::Complex;

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

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let (r, p_val) = stats::pearsonr(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-9);
        assert!(p_val < 0.05);

        let (sr, sp_val) = stats::spearmanr(&x, &y).unwrap();
        assert!((sr - 1.0).abs() < 1e-9);
        assert!(sp_val < 0.05);
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

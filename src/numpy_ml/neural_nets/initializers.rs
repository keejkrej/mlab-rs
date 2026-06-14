use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Calculate fan-in and fan-out for a weight tensor.
pub fn calc_fan(weight_shape: (usize, usize)) -> (usize, usize) {
    let (fan_in, fan_out) = weight_shape;
    (fan_in, fan_out)
}

/// Glorot uniform initialization.
/// Draws from Uniform(-b, b) where b = gain * sqrt(6 / (fan_in + fan_out)).
pub fn glorot_uniform(rows: usize, cols: usize, gain: f64) -> Array2<f64> {
    let (fan_in, fan_out) = calc_fan((rows, cols));
    let b = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-b..b))
}

/// Glorot normal initialization.
/// Draws from TruncatedNormal(0, std) where std = gain * sqrt(2 / (fan_in + fan_out)).
pub fn glorot_normal(rows: usize, cols: usize, gain: f64) -> Array2<f64> {
    let (fan_in, fan_out) = calc_fan((rows, cols));
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    truncated_normal(0.0, std, (rows, cols))
}

/// He uniform initialization.
/// Draws from Uniform(-b, b) where b = sqrt(6 / fan_in).
pub fn he_uniform(rows: usize, cols: usize) -> Array2<f64> {
    let (fan_in, _) = calc_fan((rows, cols));
    let b = (6.0 / fan_in as f64).sqrt();
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-b..b))
}

/// He normal initialization.
/// Draws from TruncatedNormal(0, std) where std = sqrt(2 / fan_in).
pub fn he_normal(rows: usize, cols: usize) -> Array2<f64> {
    let (fan_in, _) = calc_fan((rows, cols));
    let std = (2.0 / fan_in as f64).sqrt();
    truncated_normal(0.0, std, (rows, cols))
}

/// Truncated normal distribution via rejection sampling.
pub fn truncated_normal(mean: f64, std: f64, shape: (usize, usize)) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, std).unwrap();

    let len = shape.0 * shape.1;
    let samples: Vec<f64> = (0..len)
        .map(|_| loop {
            let s = normal.sample(&mut rng);
            if (s - mean).abs() <= 2.0 * std {
                break s;
            }
        })
        .collect();

    Array2::from_shape_vec(shape, samples).unwrap()
}

/// Calculate the gain for Glorot initialization based on activation function name.
pub fn calc_glorot_gain(act_fn_name: &str) -> f64 {
    match act_fn_name {
        "Tanh" => 5.0 / 3.0,
        "ReLU" => std::f64::consts::SQRT_2,
        name if name.contains("LeakyReLU") => {
            let alpha = if let Some(start) = name.find("alpha=") {
                let rest = &name[start + 6..];
                if let Some(end) = rest.find(')') {
                    rest[..end].parse::<f64>().unwrap_or(0.3)
                } else {
                    0.3
                }
            } else {
                0.3
            };
            (2.0 / (1.0 + alpha * alpha)).sqrt()
        }
        _ => 1.0,
    }
}

/// Calculate padding dimensions for a 2D convolution.
pub fn calc_pad_dims_2d(
    in_rows: usize,
    in_cols: usize,
    out_rows: usize,
    out_cols: usize,
    kernel_shape: (usize, usize),
    stride: usize,
    dilation: usize,
) -> (usize, usize, usize, usize) {
    let d = dilation;
    let (fr, fc) = kernel_shape;
    let _fr = fr * (d + 1) - d;
    let _fc = fc * (d + 1) - d;

    let pr = (stride * (out_rows - 1) + _fr - in_rows) / 2;
    let pc = (stride * (out_cols - 1) + _fc - in_cols) / 2;

    let pr1 = pr;
    let mut pr2 = pr;
    let pc1 = pc;
    let mut pc2 = pc;

    let out_rows1 = 1 + (in_rows + 2 * pr - _fr) / stride;
    let out_cols1 = 1 + (in_cols + 2 * pc - _fc) / stride;

    if out_rows1 == out_rows.wrapping_sub(1) {
        pr2 = pr + 1;
    }
    if out_cols1 == out_cols.wrapping_sub(1) {
        pc2 = pc + 1;
    }

    (pr1, pr2, pc1, pc2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_fan_2d() {
        let (fan_in, fan_out) = calc_fan((128, 64));
        assert_eq!(fan_in, 128);
        assert_eq!(fan_out, 64);
    }

    #[test]
    fn test_glorot_uniform() {
        let w = glorot_uniform(100, 50, 1.0);
        assert_eq!(w.dim(), (100, 50));
    }

    #[test]
    fn test_he_normal() {
        let w = he_normal(100, 50);
        assert_eq!(w.dim(), (100, 50));
    }

    #[test]
    fn test_truncated_normal() {
        let w = truncated_normal(0.0, 1.0, (100, 50));
        assert_eq!(w.dim(), (100, 50));
        for &v in w.iter() {
            assert!(v.abs() <= 2.5);
        }
    }
}

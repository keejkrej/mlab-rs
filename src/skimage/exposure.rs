use ndarray::Array2;

/// Histogram equalization for contrast enhancement.
pub fn equalize_hist(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut hist = [0usize; 256];
    for y in 0..height {
        for x in 0..width {
            hist[image[[y, x]] as usize] += 1;
        }
    }
    let total = (height * width) as usize;
    let mut cdf = 0usize;
    let mut mapping = [0u8; 256];
    for (i, count) in hist.iter().enumerate() {
        cdf += *count;
        let value = ((cdf as f64 * 255.0) / total as f64).round() as u8;
        mapping[i] = value;
    }

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = mapping[image[[y, x]] as usize];
        }
    }
    out
}

/// Rescale intensity values from one range to another.
pub fn rescale_intensity(image: &Array2<f64>, in_range: (f64, f64), out_range: (f64, f64)) -> Array2<f64> {
    let (height, width) = image.dim();
    let (in_min, in_max) = in_range;
    let (out_min, out_max) = out_range;
    let input_span = (in_max - in_min).abs();
    let output_span = out_max - out_min;

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let v = image[[y, x]];
            let scaled = if input_span < 1e-12 {
                0.0
            } else {
                (v - in_min) / input_span
            };
            out[[y, x]] = out_min + scaled * output_span;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equalize_hist_and_rescale() {
        let image = Array2::from_shape_vec((2, 2), vec![0u8, 255, 0, 255]).unwrap();
        let equalized = equalize_hist(&image);
        assert_eq!(equalized.dim(), (2, 2));
        assert!(equalized.iter().all(|&v| v <= u8::MAX));

        let scaled = rescale_intensity(&Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap(), (0.0, 3.0), (0.0, 1.0));
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-9);
        assert!((scaled[[1, 1]] - 1.0).abs() < 1e-9);
    }
}

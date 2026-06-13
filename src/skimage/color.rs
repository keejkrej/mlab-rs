use ndarray::{Array2, Array3};

/// Convert RGB image (H, W, 3) to grayscale (H, W) using standard luminosity weights.
pub fn rgb2gray(rgb: &Array3<u8>) -> Array2<u8> {
    let (height, width, channels) = rgb.dim();
    assert_eq!(channels, 3, "Input array must have 3 channels");
    let mut gray = Array2::zeros((height, width));
    for r in 0..height {
        for c in 0..width {
            let r_val = rgb[[r, c, 0]] as f64;
            let g_val = rgb[[r, c, 1]] as f64;
            let b_val = rgb[[r, c, 2]] as f64;
            let gray_val = (0.299 * r_val + 0.587 * g_val + 0.114 * b_val).round() as u8;
            gray[[r, c]] = gray_val;
        }
    }
    gray
}

/// Convert grayscale image (H, W) to RGB (H, W, 3).
pub fn gray2rgb(gray: &Array2<u8>) -> Array3<u8> {
    let (height, width) = gray.dim();
    let mut rgb = Array3::zeros((height, width, 3));
    for r in 0..height {
        for c in 0..width {
            let val = gray[[r, c]];
            rgb[[r, c, 0]] = val;
            rgb[[r, c, 1]] = val;
            rgb[[r, c, 2]] = val;
        }
    }
    rgb
}

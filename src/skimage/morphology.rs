use ndarray::Array2;

/// Binary erosion on a 2D binary image.
pub fn binary_erosion(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut all_one = true;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if image[[ny, nx]] == 0 {
                        all_one = false;
                    }
                }
            }
            out[[y, x]] = if all_one { 1 } else { 0 };
        }
    }
    out
}

/// Binary dilation on a 2D binary image.
pub fn binary_dilation(image: &Array2<u8>) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut any_one = false;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if image[[ny, nx]] == 1 {
                        any_one = true;
                    }
                }
            }
            out[[y, x]] = if any_one { 1 } else { 0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_ops() {
        let image = Array2::from_shape_vec((3, 3), vec![1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let eroded = binary_erosion(&image);
        let dilated = binary_dilation(&image);
        assert_eq!(eroded.dim(), (3, 3));
        assert_eq!(dilated.dim(), (3, 3));
        assert_eq!(eroded[[1, 1]], 1);
        assert_eq!(dilated[[1, 1]], 1);
    }
}

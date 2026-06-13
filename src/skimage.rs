pub mod color;
pub mod io;
pub mod filters;
pub mod transform;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_color() {
        let mut rgb = Array3::zeros((10, 10, 3));
        for y in 0..10 {
            for x in 0..10 {
                rgb[[y, x, 0]] = 255;
            }
        }
        let gray = color::rgb2gray(&rgb);
        assert_eq!(gray.dim(), (10, 10));
        assert_eq!(gray[[0, 0]], 76);

        let rgb_back = color::gray2rgb(&gray);
        assert_eq!(rgb_back.dim(), (10, 10, 3));
        assert_eq!(rgb_back[[0, 0, 0]], 76);
        assert_eq!(rgb_back[[0, 0, 1]], 76);
        assert_eq!(rgb_back[[0, 0, 2]], 76);
    }

    #[test]
    fn test_filters() {
        let image = Array2::zeros((8, 8));
        let blurred = filters::gaussian(&image, 1.0);
        assert_eq!(blurred.dim(), (8, 8));

        let edges = filters::sobel(&image);
        assert_eq!(edges.dim(), (8, 8));

        let canny_edges = filters::canny(&image, 50.0, 150.0);
        assert_eq!(canny_edges.dim(), (8, 8));
    }

    #[test]
    fn test_transform() {
        let image = Array3::zeros((2, 2, 3));
        let resized = transform::resize(&image, (4, 4));
        assert_eq!(resized.dim(), (4, 4, 3));
    }
}

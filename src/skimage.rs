use ndarray::{Array2, Array3};

// --- color submodule ---

pub mod color {
    use super::*;

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
}

// --- io submodule ---

pub mod io {
    use super::*;
    use image::{DynamicImage, GenericImageView, ImageReader};

    /// Load an image from a file path into an RGB array (H, W, 3).
    pub fn imread(path: &str) -> Result<Array3<u8>, String> {
        let img = ImageReader::open(path)
            .map_err(|e| e.to_string())?
            .decode()
            .map_err(|e| e.to_string())?;

        let (width, height) = img.dimensions();
        let rgb = img.to_rgb8();
        let mut arr = Array3::zeros((height as usize, width as usize, 3));
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                arr[[y as usize, x as usize, 0]] = pixel[0];
                arr[[y as usize, x as usize, 1]] = pixel[1];
                arr[[y as usize, x as usize, 2]] = pixel[2];
            }
        }
        Ok(arr)
    }

    /// Save an RGB array (H, W, 3) to a file path.
    pub fn imsave(path: &str, arr: &Array3<u8>) -> Result<(), String> {
        let (height, width, channels) = arr.dim();
        assert_eq!(channels, 3, "Image must have 3 channels");
        let mut img = image::ImageBuffer::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let r = arr[[y, x, 0]];
                let g = arr[[y, x, 1]];
                let b = arr[[y, x, 2]];
                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img).save(path).map_err(|e| e.to_string())
    }
}

// --- filters submodule ---

pub mod filters {
    use super::*;
    use image::{GrayImage, Luma};

    /// Apply Gaussian blur to a grayscale image.
    pub fn gaussian(image: &Array2<u8>, sigma: f64) -> Array2<u8> {
        let (height, width) = image.dim();
        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
            }
        }

        let blurred = image::imageops::blur(&gray_img, sigma as f32);

        let mut out = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                out[[y, x]] = blurred.get_pixel(x as u32, y as u32)[0];
            }
        }
        out
    }

    /// Apply Sobel filter to find edges on a grayscale image.
    pub fn sobel(image: &Array2<u8>) -> Array2<u8> {
        let (height, width) = image.dim();
        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
            }
        }

        let grad = imageproc::gradients::sobel_gradients(&gray_img);

        let mut out = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                out[[y, x]] = grad.get_pixel(x as u32, y as u32)[0].min(255) as u8;
            }
        }
        out
    }
}

// --- transform submodule ---

pub mod transform {
    use super::*;
    use image::{ImageBuffer, Rgb};

    /// Resize an RGB image to a new shape (H, W).
    pub fn resize(image: &Array3<u8>, output_shape: (usize, usize)) -> Array3<u8> {
        let (height, width, channels) = image.dim();
        assert_eq!(channels, 3, "Only 3-channel RGB image resize is supported");

        let mut img_buf = ImageBuffer::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let r = image[[y, x, 0]];
                let g = image[[y, x, 1]];
                let b = image[[y, x, 2]];
                img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        let resized = image::imageops::resize(
            &img_buf,
            output_shape.1 as u32,
            output_shape.0 as u32,
            image::imageops::FilterType::Triangle,
        );

        let mut out = Array3::zeros((output_shape.0, output_shape.1, 3));
        for y in 0..output_shape.0 {
            for x in 0..output_shape.1 {
                let pixel = resized.get_pixel(x as u32, y as u32);
                out[[y, x, 0]] = pixel[0];
                out[[y, x, 1]] = pixel[1];
                out[[y, x, 2]] = pixel[2];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn test_transform() {
        let image = Array3::zeros((2, 2, 3));
        let resized = transform::resize(&image, (4, 4));
        assert_eq!(resized.dim(), (4, 4, 3));
    }
}


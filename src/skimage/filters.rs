use ndarray::Array2;
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

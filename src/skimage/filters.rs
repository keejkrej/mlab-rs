use ndarray::Array2;
use image::{GrayImage, Luma};

/// Compute Otsu's optimal threshold for a grayscale image.
pub fn threshold_otsu(image: &Array2<u8>) -> f64 {
    let (height, width) = image.dim();
    let total = (height * width) as usize;
    let mut hist = [0usize; 256];
    for y in 0..height {
        for x in 0..width {
            hist[image[[y, x]] as usize] += 1;
        }
    }
    let mut sum = 0.0;
    for i in 0..256 {
        sum += i as f64 * hist[i] as f64;
    }
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_var = 0.0;
    let mut threshold = 0.0;
    for i in 0..256 {
        w_b += hist[i] as f64;
        if w_b <= 0.0 || w_b >= total as f64 { continue; }
        let w_f = total as f64 - w_b;
        if w_f <= 0.0 { break; }
        sum_b += i as f64 * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > max_var {
            max_var = var_between;
            threshold = i as f64;
        }
    }
    threshold
}

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

/// Apply Canny edge detection filter to a grayscale image.
pub fn canny(image: &Array2<u8>, low_threshold: f32, high_threshold: f32) -> Array2<u8> {
    let (height, width) = image.dim();
    let mut gray_img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            gray_img.put_pixel(x as u32, y as u32, Luma([image[[y, x]]]));
        }
    }

    let edges = imageproc::edges::canny(&gray_img, low_threshold, high_threshold);

    let mut out = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = edges.get_pixel(x as u32, y as u32)[0];
        }
    }
    out
}


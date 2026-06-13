use ndarray::Array3;
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

use ndarray::Array3;
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

use mlab_rs::np;
use mlab_rs::skimage::{color, filters, transform};

fn main() {
    println!("--- Scikit-Image Demo ---");

    // Create a mock RGB image: 4x4 with 3 channels
    let mut image = np::zeros((4, 4, 3));
    // Set some colors
    image[[0, 0, 0]] = 255; // Red pixel at (0,0)
    image[[1, 1, 1]] = 255; // Green pixel at (1,1)
    image[[2, 2, 2]] = 255; // Blue pixel at (2,2)
    println!("Original mock RGB image (red channel):\n{:?}", image.slice(np::s![.., .., 0]));

    // Convert to grayscale: rgb2gray
    // Python: gray = color.rgb2gray(image)
    let gray = color::rgb2gray(&image);
    println!("Grayscale image:\n{:?}", gray);

    // Apply filters: gaussian blur
    // Python: blurred = filters.gaussian(gray, sigma=1.0)
    let blurred = filters::gaussian(&gray, 1.0);
    println!("Gaussian blurred image:\n{:?}", blurred);

    // Apply Canny edge detection
    // Python: edges = filters.canny(gray, low_threshold=50.0, high_threshold=150.0)
    let edges = filters::canny(&gray, 50.0, 150.0);
    println!("Canny edges:\n{:?}", edges);

    // Resize image
    // Python: resized = transform.resize(image, (8, 8))
    let resized = transform::resize(&image, (8, 8));
    println!("Resized image shape: {:?}", resized.dim());
}

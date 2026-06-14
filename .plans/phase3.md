# Phase 3: Signal & Image Processing

Signal processing, interpolation, and image analysis.

## sp.signal -- filter design and spectral analysis

- [ ] `butter` -- Butterworth filter design (returns sos/zpk)
- [ ] `cheby1` -- Chebyshev Type I filter design
- [ ] `cheby2` -- Chebyshev Type II filter design
- [ ] `ellip` -- elliptic filter design
- [ ] `sosfilt` -- second-order sections filtering
- [ ] `sosfiltfilt` -- zero-phase SOS filtering
- [ ] `correlate` -- cross-correlation (1D, 3 modes)
- [ ] `fftconvolve` -- FFT-based convolution
- [ ] `resample` -- resample signal to new length
- [ ] `resample_poly` -- resample using polyphase filtering
- [ ] `welch` -- power spectral density via Welch's method
- [ ] `spectrogram` -- spectrogram via STFT
- [ ] `stft` / `istft` -- short-time Fourier transform
- [ ] `freqz` -- frequency response of digital filter
- [ ] `savgol_filter` -- Savitzky-Golay filter
- [ ] `medfilt` -- median filter (1D)
- [ ] `detrend` -- remove linear or constant trend
- [ ] `hilbert` / `hilbert2` -- analytic signal via Hilbert transform
- [ ] `peak_widths` -- widths of peaks at given prominence
- [ ] `peak_prominences` -- prominences of peaks

## sp.ndimage -- N-D image processing

- [ ] `gaussian_filter` -- N-D Gaussian filter
- [ ] `gaussian_filter1d` -- 1D Gaussian filter
- [ ] `uniform_filter` -- N-D uniform (box) filter
- [ ] `median_filter` -- N-D median filter
- [ ] `maximum_filter` / `minimum_filter` -- N-D min/max filter
- [ ] `binary_erosion` -- binary morphological erosion
- [ ] `binary_dilation` -- binary morphological dilation
- [ ] `binary_opening` / `binary_closing` -- binary morphological open/close
- [ ] `grey_erosion` / `grey_dilation` -- grayscale morphology
- [ ] `label` -- connected component labeling
- [ ] `find_objects` -- find bounding boxes of labeled regions
- [ ] `center_of_mass` -- center of mass of labeled regions
- [ ] `sum_labels` / `mean_labels` -- per-label statistics
- [ ] `zoom` -- N-D zoom/resampling
- [ ] `rotate` -- N-D rotation
- [ ] `shift` -- N-D shift
- [ ] `affine_transform` -- N-D affine transformation
- [ ] `map_coordinates` -- interpolate at arbitrary coordinates
- [ ] `distance_transform_edt` -- Euclidean distance transform
- [ ] `sobel` / `prewitt` / `laplace` -- edge detection filters
- [ ] `convolve` / `correlate` -- N-D convolution/correlation

## sp.interpolate -- additional interpolation methods

- [ ] `CubicSpline` -- natural/clamped cubic spline
- [ ] `interp1d` -- scipy-compatible 1D interpolation
- [ ] `RBFInterpolator` -- radial basis function interpolation
- [ ] `griddata` -- unstructured D-dimensional interpolation
- [ ] `RegularGridInterpolator` -- interpolation on regular grid
- [ ] `Akima1DInterpolator` -- Akima interpolation
- [ ] `UnivariateSpline` -- smoothing spline
- [ ] `BSpline` -- B-spline representation
- [ ] `PPoly` -- piecewise polynomial

## skimage.measure -- image measurement

- [ ] `regionprops` -- region properties (area, bbox, centroid, moments)
- [ ] `find_contours` -- iso-valued contours
- [ ] `label` -- connected component labeling (2D)
- [ ] `moments` / `moments_central` / `moments_hu` -- image moments
- [ ] `centroid` -- region centroid
- [ ] `perimeter` -- region perimeter
- [ ] `inertia_tensor` -- inertia tensor of labeled region

## skimage.metrics -- image quality metrics

- [ ] `PSNR` -- peak signal-to-noise ratio
- [ ] `structural_similarity` (SSIM) -- structural similarity index
- [ ] `mean_squared_error` -- image MSE

## skimage.morphology -- additional morphology

- [ ] `binary_opening` / `binary_closing` -- binary morphological ops
- [ ] `erosion` / `dilation` -- grayscale morphological ops
- [ ] `opening` / `closing` -- grayscale morphological ops
- [ ] `skeletonize` -- skeletonization
- [ ] `remove_small_objects` -- remove small connected components
- [ ] `disk` / `square` / `diamond` -- structuring element generators
- [ ] `label` -- connected component labeling
- [ ] `watershed` -- watershed segmentation

## skimage.filters -- additional filters

- [ ] `laplace` -- Laplacian filter
- [ ] `median` -- median filter
- [ ] `threshold_local` -- local adaptive thresholding
- [ ] `threshold_yen` / `threshold_li` / `threshold_triangle` -- additional threshold methods
- [ ] `unsharp_mask` -- unsharp masking
- [ ] `prewitt` / `scharr` -- additional edge detectors
- [ ] `difference_of_gaussians` -- DoG filter

## skimage.exposure -- additional exposure ops

- [ ] `equalize_adapthist` (CLAHE) -- contrast-limited adaptive histogram equalization
- [ ] `adjust_gamma` -- gamma correction
- [ ] `adjust_log` -- logarithmic adjustment
- [ ] `histogram` -- image histogram

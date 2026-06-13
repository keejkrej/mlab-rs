use ndarray::Array1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvolveMode {
    Full,
    Same,
    Valid,
}

/// 1D Convolution of two signals.
pub fn convolve(in1: &Array1<f64>, in2: &Array1<f64>, mode: ConvolveMode) -> Array1<f64> {
    let n1 = in1.len();
    let n2 = in2.len();
    if n1 == 0 || n2 == 0 {
        return Array1::from_vec(vec![]);
    }

    let out_len = n1 + n2 - 1;
    let mut full_result = vec![0.0; out_len];

    for i in 0..n1 {
        for j in 0..n2 {
            full_result[i + j] += in1[i] * in2[j];
        }
    }

    match mode {
        ConvolveMode::Full => Array1::from_vec(full_result),
        ConvolveMode::Same => {
            let start = (n2 - 1) / 2;
            let end = start + n1;
            Array1::from_vec(full_result[start..end].to_vec())
        }
        ConvolveMode::Valid => {
            if n1 >= n2 {
                let start = n2 - 1;
                let end = n1;
                Array1::from_vec(full_result[start..end].to_vec())
            } else {
                let start = n1 - 1;
                let end = n2;
                Array1::from_vec(full_result[start..end].to_vec())
            }
        }
    }
}

/// Filter data along one dimension with an IIR or FIR filter using Transposed Direct Form II.
pub fn lfilter(b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Result<Array1<f64>, String> {
    if a.len() == 0 || b.len() == 0 {
        return Err("Filter coefficients a and b must not be empty".to_string());
    }
    let a0 = a[0];
    if a0.abs() < 1e-14 {
        return Err("First denominator coefficient a[0] must not be zero".to_string());
    }

    let b_norm: Vec<f64> = b.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&x| x / a0).collect();

    let n = x.len();
    let mut y = Array1::zeros(n);

    let n_b = b_norm.len();
    let n_a = a_norm.len();
    let ord = std::cmp::max(n_b, n_a);
    let mut z = vec![0.0; ord + 1];

    for i in 0..n {
        let xi = x[i];
        let yi = b_norm[0] * xi + z[0];
        y[i] = yi;

        for k in 1..ord {
            let bk = if k < n_b { b_norm[k] } else { 0.0 };
            let ak = if k < n_a { a_norm[k] } else { 0.0 };
            z[k - 1] = bk * xi - ak * yi + z[k];
        }
    }

    Ok(y)
}

/// Zero-phase forward-backward digital filter.
pub fn filtfilt(b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Result<Array1<f64>, String> {
    let padlen = 3 * std::cmp::max(b.len(), a.len());
    if x.len() <= padlen {
        return Err("Input signal is too short for the requested padlen".to_string());
    }

    // Construct padded signal with reflection padding
    let mut padded_vec = Vec::with_capacity(x.len() + 2 * padlen);

    let x0 = x[0];
    for i in (1..=padlen).rev() {
        padded_vec.push(2.0 * x0 - x[i]);
    }

    for &val in x.iter() {
        padded_vec.push(val);
    }

    let xn = x[x.len() - 1];
    let n = x.len();
    for i in 1..=padlen {
        padded_vec.push(2.0 * xn - x[n - 1 - i]);
    }

    let padded = Array1::from_vec(padded_vec);

    // Forward filter
    let y1 = lfilter(b, a, &padded)?;

    // Reverse y1
    let mut y1_rev = y1.to_vec();
    y1_rev.reverse();
    let y1_rev_arr = Array1::from_vec(y1_rev);

    // Backward filter
    let y2 = lfilter(b, a, &y1_rev_arr)?;

    // Reverse y2 back
    let mut y2_rev = y2.to_vec();
    y2_rev.reverse();

    // Crop the padded boundaries
    let start_idx = padlen;
    let end_idx = padlen + x.len();
    let cropped = y2_rev[start_idx..end_idx].to_vec();

    Ok(Array1::from_vec(cropped))
}

/// Find local peaks (maxima) in a 1D signal.
pub fn find_peaks(
    x: &Array1<f64>,
    height: Option<f64>,
    distance: Option<usize>,
) -> Vec<usize> {
    let n = x.len();
    if n < 3 {
        return vec![];
    }

    let mut peaks = Vec::new();
    for i in 1..(n - 1) {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            if let Some(h) = height {
                if x[i] < h {
                    continue;
                }
            }
            peaks.push(i);
        }
    }

    if let Some(dist) = distance {
        let mut kept = Vec::new();
        // Sort peaks by height descending
        let mut peak_heights: Vec<(usize, f64)> = peaks.iter().map(|&idx| (idx, x[idx])).collect();
        peak_heights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, _h) in peak_heights {
            let mut ok = true;
            for &k_idx in &kept {
                let diff = if idx > k_idx { idx - k_idx } else { k_idx - idx };
                if diff < dist {
                    ok = false;
                    break;
                }
            }
            if ok {
                kept.push(idx);
            }
        }
        kept.sort();
        peaks = kept;
    }

    peaks
}


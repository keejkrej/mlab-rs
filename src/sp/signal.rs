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


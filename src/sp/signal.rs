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

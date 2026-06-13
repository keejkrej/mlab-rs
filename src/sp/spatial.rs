use ndarray::Array2;

/// Compute a pairwise distance matrix between two sets of points.
pub fn rscdist(xa: &Array2<f64>, xb: &Array2<f64>, metric: &str) -> Array2<f64> {
    let (n_a, n_features) = xa.dim();
    let (n_b, _) = xb.dim();
    assert_eq!(n_features, xb.ncols(), "Feature counts must match");

    let mut out = Array2::zeros((n_a, n_b));
    for i in 0..n_a {
        for j in 0..n_b {
            let mut dist = 0.0;
            for k in 0..n_features {
                let diff = xa[[i, k]] - xb[[j, k]];
                match metric {
                    "manhattan" => dist += diff.abs(),
                    "cosine" => dist += diff * diff,
                    _ => dist += diff * diff,
                }
            }
            out[[i, j]] = match metric {
                "manhattan" => dist,
                "cosine" => {
                    let norm_a = xa.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
                    let norm_b = xb.row(j).iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm_a < 1e-12 || norm_b < 1e-12 {
                        0.0
                    } else {
                        1.0 - (xa.row(i).iter().zip(xb.row(j).iter()).map(|(u, v)| u * v).sum::<f64>() / (norm_a * norm_b))
                    }
                }
                _ => dist.sqrt(),
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rscdist() {
        let xa = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let xb = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let d = rscdist(&xa, &xb, "euclidean");
        assert_eq!(d.dim(), (2, 2));
        assert!((d[[0, 0]] - 1.0).abs() < 1e-9);
    }
}

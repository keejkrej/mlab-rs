use ndarray::Array1;

/// Standard error function approximation (Abramowitz and Stegun).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp());
    sign * y
}

pub struct Norm;

impl Norm {
    /// Probability density function of normal distribution.
    pub fn pdf(x: f64, loc: f64, scale: f64) -> f64 {
        let variance = scale * scale;
        let exponent = -((x - loc).powi(2)) / (2.0 * variance);
        (1.0 / (scale * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Cumulative distribution function of normal distribution.
    pub fn cdf(x: f64, loc: f64, scale: f64) -> f64 {
        0.5 * (1.0 + erf((x - loc) / (scale * 2.0_f64.sqrt())))
    }
}

/// Compute Pearson correlation coefficient and p-value.
pub fn pearsonr(x: &Array1<f64>, y: &Array1<f64>) -> Result<(f64, f64), String> {
    let n = x.len();
    if n != y.len() {
        return Err("Arrays must have the same length".to_string());
    }
    if n == 0 {
        return Err("Arrays must not be empty".to_string());
    }
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den < 1e-14 {
        return Ok((0.0, 1.0));
    }
    let r = num / den;

    let p_value = if n <= 2 {
        1.0
    } else if (1.0 - r * r) <= 1e-14 {
        0.0
    } else {
        let t_stat = r * ((n - 2) as f64 / (1.0 - r * r)).sqrt();
        let norm_cdf = Norm::cdf(t_stat.abs(), 0.0, 1.0);
        (2.0 * (1.0 - norm_cdf)).min(1.0).max(0.0)
    };

    Ok((r, p_value))
}

/// Compute ranks for data, handling ties by averaging ranks.
fn rankdata(arr: &Array1<f64>) -> Array1<f64> {
    let n = arr.len();
    let mut indexed: Vec<(usize, f64)> = arr.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let rank_sum: f64 = (i + 1..=j).map(|r| r as f64).sum();
        let avg_rank = rank_sum / ((j - i) as f64);
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    Array1::from_vec(ranks)
}

/// Compute Spearman rank correlation coefficient and p-value.
pub fn spearmanr(x: &Array1<f64>, y: &Array1<f64>) -> Result<(f64, f64), String> {
    let rx = rankdata(x);
    let ry = rankdata(y);
    pearsonr(&rx, &ry)
}

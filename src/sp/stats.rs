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

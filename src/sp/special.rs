use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Gamma family
// ---------------------------------------------------------------------------

/// Gamma function Γ(x).
///
/// Uses Lanczos approximation for x > 0 and the reflection formula for x < 0.
/// Returns `f64::NAN` for non-positive integers (poles).
pub fn gamma(x: f64) -> f64 {
    if x > 0.0 {
        gammaln(x).exp()
    } else if x == 0.0 {
        f64::INFINITY
    } else if x.fract() == 0.0 && x == x.trunc() {
        // Negative integer → pole
        f64::NAN
    } else {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        PI / ((PI * x).sin() * gamma(1.0 - x))
    }
}

/// Log of the absolute value of the Gamma function, ln|Γ(x)|.
///
/// Uses the Lanczos approximation (g=7, n=9 coefficients).
pub fn gammaln(x: f64) -> f64 {
    if x < 0.5 {
        // Use reflection: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        return PI.ln() - (PI * x).sin().abs().ln() - gammaln(1.0 - x);
    }

    let x = x - 1.0;
    // Lanczos coefficients (g=7, n=9)
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut y = c[0];
    for i in 1..9 {
        y += c[i] / (x + i as f64);
    }

    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + y.ln()
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence relation to shift x into the asymptotic region,
/// then applies the series expansion.
pub fn digamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // Use recurrence to shift x > 8
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln() - 0.5 * inv_x
        - inv_x2 / 12.0 * (1.0
            - inv_x2 / 10.0 * (1.0 - inv_x2 / 42.0 * (1.0 - inv_x2 / 40.0)));

    result
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
pub fn beta(a: f64, b: f64) -> f64 {
    betaln(a, b).exp()
}

/// Log of the Beta function, ln B(a, b).
pub fn betaln(a: f64, b: f64) -> f64 {
    gammaln(a) + gammaln(b) - gammaln(a + b)
}

// ---------------------------------------------------------------------------
// Error functions
// ---------------------------------------------------------------------------

/// Scaled complementary error function erfcx(x) = exp(x²) * erfc(x).
///
/// Uses a rational approximation for numerical stability.
pub fn erfcx(x: f64) -> f64 {
    if x < 0.0 {
        // erfcx(-x) = 2*exp(x²) - erfcx(x)
        let pos = erfcx(-x);
        2.0 * (x * x).exp() - pos
    } else if x < 6.0 {
        // Rational approximation (Abramowitz & Stegun 7.1.26 adapted)
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erfc_val = poly * (-x * x).exp();
        erfc_val * (x * x).exp()
    } else {
        // For large x, use asymptotic: erfcx(x) ≈ 1/(sqrt(π)*x)
        let inv_sqrt_pi = 1.0 / PI.sqrt();
        inv_sqrt_pi / x * (1.0 - 0.5 / (x * x) + 0.75 / (x * x * x * x))
    }
}

/// Inverse error function.
///
/// Returns x such that erf(x) = p, for -1 < p < 1.
/// Uses a rational approximation.
pub fn erfinv(p: f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p <= -1.0 {
        return f64::NEG_INFINITY;
    }

    let sign = if p < 0.0 { -1.0 } else { 1.0 };
    let p = p.abs();

    // Initial guess using rational approximation
    // Based on the approximation from Giles (2010)
    let a = 0.147;
    let ln1p2 = (1.0 - p * p).ln();
    let tt1 = 2.0 / (std::f64::consts::PI * a) + ln1p2 / 2.0;
    let tt2 = ln1p2 / a;
    let mut x = sign * (-tt1 + (tt1 * tt1 - tt2).sqrt()).sqrt();

    // Newton-Raphson refinement: solve erf(x) = p
    // x_{n+1} = x_n - (erf(x_n) - p) / (2/sqrt(pi) * exp(-x_n^2))
    let sqrt_pi = std::f64::consts::PI.sqrt();
    for _ in 0..5 {
        let erf_x = erf(x);
        let deriv = (2.0 / sqrt_pi) * (-x * x).exp();
        x -= (erf_x - p) / deriv;
    }

    sign * x
}

/// Error function using Horner form rational approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz & Stegun 7.1.26 (max error 1.5e-7)
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
        - 1.453152027 * t4
        + 1.061405429 * t5;

    sign * (1.0 - poly * (-x * x).exp())
}

/// Inverse of the standard normal CDF (rational approximation).
fn norminv(p: f64) -> f64 {
    // Peter Acklam's algorithm for inverse normal CDF
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for the rational approximation (Acklam)
    let a1 = -3.969683028665376e+01;
    let a2 = 2.209460984245205e+02;
    let a3 = -2.759285104469687e+02;
    let a4 = 1.383577518672690e+02;
    let a5 = -3.066479806614716e+01;
    let a6 = 2.506628277459239e+00;

    let b1 = -5.447609879822406e+01;
    let b2 = 1.615858368580409e+02;
    let b3 = -1.556989798598866e+02;
    let b4 = 6.680131188771972e+01;
    let b5 = -1.328068155288572e+01;

    let c1 = -7.784894002430293e-03;
    let c2 = -3.223964580411365e-01;
    let c3 = -2.400758277161838e+00;
    let c4 = -2.549732539343734e+00;
    let c5 = 4.374664141464968e+00;
    let c6 = 2.938163982698783e+00;

    let d1 = 7.784695709041462e-03;
    let d2 = 3.224671290700398e-01;
    let d3 = 2.445134137142996e+00;
    let d4 = 3.754408661907416e+00;

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let (q, r) = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        let r = a1 + q * (a2 + q * (a3 + q * (a4 + q * (a5 + q * a6))));
        let s = 1.0 + q * (b1 + q * (b2 + q * (b3 + q * (b4 + q * b5))));
        (q, r / s)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        let num = a1 + r * (a2 + r * (a3 + r * (a4 + r * (a5 + r * a6))));
        let den = 1.0 + r * (b1 + r * (b2 + r * (b3 + r * (b4 + r * b5))));
        (q, q * num / den)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let r = c1 + q * (c2 + q * (c3 + q * (c4 + q * (c5 + q * c6))));
        let s = 1.0 + q * (d1 + q * (d2 + q * (d3 + q * d4)));
        (q, -r / s)
    };

    q + r
}

// ---------------------------------------------------------------------------
// Logistic functions
// ---------------------------------------------------------------------------

/// Logistic sigmoid (expit) function: 1 / (1 + exp(-x)).
///
/// Numerically stable for all inputs.
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Logit function: log(p / (1 - p)).
///
/// Returns `f64::NAN` for p outside (0, 1).
pub fn logit(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        f64::NAN
    } else {
        (p / (1.0 - p)).ln()
    }
}

// ---------------------------------------------------------------------------
// Numerically stable operations
// ---------------------------------------------------------------------------

/// Log of sum of exponentials: log(sum(exp(x_i))).
///
/// Computed in a numerically stable way by subtracting the max value first.
pub fn logsumexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::INFINITY {
        return f64::INFINITY;
    }
    let sum: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Numerically stable softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j)).
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Bessel functions
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind I₀(x).
///
/// Uses the Chebyshev approximation from Numerical Recipes.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Modified Bessel function of the first kind I₁(x).
///
/// Uses the Chebyshev approximation from Numerical Recipes.
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        sign * ax / 2.0
            * (0.5
                + y * (0.87890594
                    + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
    } else {
        let y = 3.75 / ax;
        sign * (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (-0.03988024
                    + y * (-0.00362018
                        + y * (0.00163801
                            + y * (-0.01031555
                                + y * (0.02282967
                                    + y * (-0.02895312 + y * (0.01787654 + y * (-0.00420059)))))))))
    }
}

/// Bessel function of the first kind J₀(x).
///
/// Uses a rational approximation for different ranges.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        // Rational approximation (Abramowitz & Stegun 9.4.1)
        let num = 57568490574.0
            + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;

        let f0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let g0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * 0.7621095161e-6));

        (2.0 / PI / ax).sqrt() * (f0 * xx.cos() - z * g0 * xx.sin())
    }
}

/// Bessel function of the first kind J₁(x).
///
/// Uses a rational approximation for different ranges.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72362614232.0
                + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let den = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;

        let f1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let g1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6)));

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (2.0 / PI / ax).sqrt() * (f1 * xx.cos() - z * g1 * xx.sin())
    }
}

// ---------------------------------------------------------------------------
// Combinatorial
// ---------------------------------------------------------------------------

/// Factorial n! as f64.
///
/// Handles n up to 170 (the f64 overflow limit). Returns `f64::INFINITY` for n > 170.
pub fn factorial(n: u64) -> f64 {
    if n > 170 {
        return f64::INFINITY;
    }
    let mut result = 1.0_f64;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Log of factorial: ln(n!).
///
/// Uses `gammaln` for numerical stability.
pub fn factorialln(n: u64) -> f64 {
    gammaln(n as f64 + 1.0)
}

/// Binomial coefficient C(n, k) as f64.
pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use symmetry to minimize operations
    let k = if k > n - k { n - k } else { k };
    (factorialln(n) - factorialln(k) - factorialln(n - k)).exp()
}

/// Permutation P(n, k) = n! / (n - k)! as f64.
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    (factorialln(n) - factorialln(n - k)).exp()
}

/// Pochhammer symbol (rising factorial) (x)_n = Γ(x + n) / Γ(x).
pub fn poch(x: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 1.0;
    }
    (gammaln(x + n) - gammaln(x)).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gamma family ---

    #[test]
    fn test_gamma_positive_integers() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-9);
        assert!((gamma(2.0) - 1.0).abs() < 1e-9);
        assert!((gamma(3.0) - 2.0).abs() < 1e-9);
        assert!((gamma(4.0) - 6.0).abs() < 1e-9);
        assert!((gamma(5.0) - 24.0).abs() < 1e-9);
        assert!((gamma(6.0) - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_gamma_half_integer() {
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-9);
        assert!((gamma(1.5) - 0.5 * PI.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_gamma_negative_half() {
        // Γ(-0.5) = -2√π
        assert!((gamma(-0.5) - (-2.0 * PI.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_gammaln() {
        assert!((gammaln(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((gammaln(1.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_digamma() {
        // ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772)
        let euler_mascheroni = 0.5772156649015329;
        assert!((digamma(1.0) + euler_mascheroni).abs() < 1e-4);
        // ψ(2) = 1 - γ
        assert!((digamma(2.0) - (1.0 - euler_mascheroni)).abs() < 1e-4);
    }

    #[test]
    fn test_beta() {
        // B(2, 3) = 1/12
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-9);
        // B(1, 1) = 1
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_betaln() {
        assert!((betaln(2.0, 3.0) - (1.0 / 12.0_f64).ln()).abs() < 1e-9);
    }

    // --- Error functions ---

    #[test]
    fn test_erfcx_zero() {
        // erfcx(0) = erfc(0) * exp(0) = 1
        assert!((erfcx(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erfinv_zero() {
        assert!((erfinv(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_erfinv_half() {
        // erfinv(0.5) ≈ 0.4769362762044699
        assert!((erfinv(0.5) - 0.4769362762044699).abs() < 1e-4);
    }

    #[test]
    fn test_erfinv_symmetry() {
        // erfinv is odd: erfinv(-x) = -erfinv(x)
        let x = 0.3;
        assert!((erfinv(-x) + erfinv(x)).abs() < 1e-9);
    }

    // --- Logistic functions ---

    #[test]
    fn test_expit_zero() {
        assert!((expit(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_expit_large_positive() {
        assert!((expit(100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_expit_large_negative() {
        assert!(expit(-100.0).abs() < 1e-9);
    }

    #[test]
    fn test_logit_half() {
        assert!((logit(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_logit_expit_roundtrip() {
        let x = 2.5;
        assert!((logit(expit(x)) - x).abs() < 1e-9);
    }

    // --- Numerically stable operations ---

    #[test]
    fn test_logsumexp_basic() {
        let x = [1.0, 2.0, 3.0];
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((logsumexp(&x) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_logsumexp_single() {
        let x = [5.0];
        assert!((logsumexp(&x) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let s = softmax(&x);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_uniform() {
        let x = [0.0, 0.0, 0.0];
        let s = softmax(&x);
        for &v in &s {
            assert!((v - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    // --- Bessel functions ---

    #[test]
    fn test_bessel_i0_zero() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_i1_zero() {
        assert!((bessel_i1(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bessel_j0_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j1_zero() {
        assert!((bessel_j1(0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bessel_j0_first_zero() {
        // First zero of J₀ is at x ≈ 2.4048255577
        let x0 = 2.4048255577;
        assert!(bessel_j0(x0).abs() < 1e-4);
    }

    // --- Combinatorial ---

    #[test]
    fn test_factorial_basic() {
        assert!((factorial(0) - 1.0).abs() < 1e-9);
        assert!((factorial(1) - 1.0).abs() < 1e-9);
        assert!((factorial(5) - 120.0).abs() < 1e-9);
        assert!((factorial(10) - 3628800.0).abs() < 1e-9);
    }

    #[test]
    fn test_factorial_large() {
        // 170! is the largest factorial representable in f64
        let f170 = factorial(170);
        assert!(f170.is_finite());
        assert!(factorial(171).is_infinite());
    }

    #[test]
    fn test_factorialln() {
        assert!((factorialln(5) - 120.0_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn test_comb() {
        assert!((comb(10, 3) - 120.0).abs() < 1e-6);
        assert!((comb(10, 0) - 1.0).abs() < 1e-9);
        assert!((comb(10, 10) - 1.0).abs() < 1e-9);
        assert!((comb(5, 3) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_perm() {
        assert!((perm(10, 3) - 720.0).abs() < 1e-6);
        assert!((perm(5, 2) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_poch() {
        // (x)_0 = 1
        assert!((poch(3.0, 0.0) - 1.0).abs() < 1e-9);
        // (2)_3 = 2*3*4 = 24
        assert!((poch(2.0, 3.0) - 24.0).abs() < 1e-6);
        // (1)_n = n!
        assert!((poch(1.0, 5.0) - 120.0).abs() < 1e-6);
    }
}

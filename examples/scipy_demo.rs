use mlab_rs::{np, sp};

fn main() {
    println!("--- SciPy Demo ---");

    // Linalg solving: A * x = B
    // Python: A = np.array([[4, 7], [2, 6]]); B = np.array([1, 2])
    let a = np::array(vec![vec![4.0, 7.0], vec![2.0, 6.0]]);
    let b = np::array(vec![1.0, 2.0]);

    // Python: x = scipy.linalg.solve(A, B)
    let x = sp::linalg::solve_vec(&a, &b).unwrap();
    println!("x (solution to A*x = B):\n{:?}\n", x);

    // Inverse of a matrix
    // Python: a_inv = scipy.linalg.inv(A)
    let a_inv = sp::linalg::inv(&a).unwrap();
    println!("A^-1 (Inverse):\n{:?}\n", a_inv);

    // FFT demo
    // Python: sig = np.array([1.0, 2.0, 1.0, 2.0])
    //         sig_fft = scipy.fft.fft(sig)
    let sig = np::array(vec![1.0, 2.0, 1.0, 2.0]);
    let sig_fft = sp::fft::fft_real(&sig);
    println!("FFT of signal:\n{:?}\n", sig_fft);

    // Stats demo (Normal distribution pdf/cdf)
    // Python: p = scipy.stats.norm.cdf(0, loc=0, scale=1)
    let p = sp::stats::Norm::cdf(0.0, 0.0, 1.0);
    println!("CDF of standard normal at 0.0: {}\n", p);

    // Correlation demo
    // Python: r, p_val = scipy.stats.pearsonr(x, y)
    let x_arr = np::array(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_arr = np::array(vec![2.0, 4.1, 5.9, 8.0, 10.1]);
    let (r, p_val) = sp::stats::pearsonr(&x_arr, &y_arr).unwrap();
    println!("Pearson correlation: r = {}, p-value = {}\n", r, p_val);

    let (sr, s_p_val) = sp::stats::spearmanr(&x_arr, &y_arr).unwrap();
    println!("Spearman rank correlation: rho = {}, p-value = {}\n", sr, s_p_val);

    // Signal convolution
    // Python: out = scipy.signal.convolve(in1, in2, mode='full')
    let in1 = np::array(vec![1.0, 2.0, 3.0]);
    let in2 = np::array(vec![0.0, 1.0, 0.5]);
    let out = sp::signal::convolve(&in1, &in2, sp::signal::ConvolveMode::Full);
    println!("Convolved output:\n{:?}\n", out);
}

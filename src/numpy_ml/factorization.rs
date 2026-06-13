use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Regularized alternating least-squares matrix factorization.
pub struct VanillaALS {
    pub k: usize,
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub w: Option<Array2<f64>>,
    pub h: Option<Array2<f64>>,
}

impl VanillaALS {
    /// Create a new ALS factorizer.
    pub fn new(k: usize, alpha: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            k,
            alpha,
            max_iter,
            tol,
            w: None,
            h: None,
        }
    }

    /// Return the current factor matrices as a tuple `(W, H)`.
    pub fn parameters(&self) -> (Option<&Array2<f64>>, Option<&Array2<f64>>) {
        (self.w.as_ref(), self.h.as_ref())
    }

    fn init_factor_matrices(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>) {
        let (n, m) = x.dim();
        let mean = x.iter().sum::<f64>() / (n * m) as f64;
        let scale = (mean / self.k as f64).sqrt();

        let mut rng = rand::thread_rng();

        self.w = w.map(|w| w.to_owned()).or_else(|| {
            Some(Array2::from_shape_fn((n, self.k), |_| rng.r#gen::<f64>() * scale))
        });
        self.h = h.map(|h| h.to_owned()).or_else(|| {
            Some(Array2::from_shape_fn((self.k, m), |_| rng.r#gen::<f64>() * scale))
        });
    }

    fn loss(&self, x: &Array2<f64>, x_hat: &Array2<f64>) -> f64 {
        let w = self.w.as_ref().unwrap();
        let h = self.h.as_ref().unwrap();
        let sq_fnorm = |a: &Array2<f64>| a.iter().map(|&v| v * v).sum::<f64>();
        sq_fnorm(&(x - x_hat)) + self.alpha * (sq_fnorm(w) + sq_fnorm(h))
    }

    fn update_factor(&self, x: &Array2<f64>, a: &Array2<f64>) -> Result<Array2<f64>, String> {
        let ata = a.t().dot(a);
        let mut reg = Array2::<f64>::eye(self.k);
        reg.mapv_inplace(|v| v * self.alpha);
        let t1 = crate::sp::linalg::inv(&(ata + reg))?;
        Ok(x.dot(a).dot(&t1))
    }

    /// Factor `x` into `W` and `H` via ALS.
    pub fn fit(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>, n_initializations: usize, verbose: bool) -> Result<(), String> {
        let n_init = if w.is_some() && h.is_some() { 1 } else { n_initializations };

        let mut best_loss = f64::INFINITY;
        let mut best_w: Option<Array2<f64>> = None;
        let mut best_h: Option<Array2<f64>> = None;

        for f in 0..n_init {
            if verbose {
                println!("\nINITIALIZATION {}", f + 1);
            }

            let (new_w, new_h, loss) = self.fit_once(x, w, h, verbose)?;

            if loss <= best_loss {
                best_loss = loss;
                best_w = Some(new_w);
                best_h = Some(new_h);
            }
        }

        self.w = best_w;
        self.h = best_h;

        if verbose {
            println!("\nFINAL LOSS: {}", best_loss);
        }

        Ok(())
    }

    fn fit_once(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>, verbose: bool) -> Result<(Array2<f64>, Array2<f64>, f64), String> {
        self.init_factor_matrices(x, w, h);
        let mut w = self.w.clone().unwrap();
        let mut h = self.h.clone().unwrap();

        for i in 0..self.max_iter {
            w = self.update_factor(x, &h.t().to_owned())?;
            h = self.update_factor(&x.t().to_owned(), &w)?.t().to_owned();

            let loss = self.loss(x, &w.dot(&h));

            if verbose {
                println!("[Iter {}] Loss: {:.8}", i + 1, loss);
            }

            if loss <= self.tol {
                break;
            }
        }

        let final_loss = self.loss(x, &w.dot(&h));
        Ok((w, h, final_loss))
    }
}

/// Nonnegative matrix factorization using hierarchical alternating least squares.
pub struct NMF {
    pub k: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub w: Option<Array2<f64>>,
    pub h: Option<Array2<f64>>,
}

impl NMF {
    /// Create a new NMF factorizer with `k` latent components.
    pub fn new(k: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            k,
            max_iter,
            tol,
            w: None,
            h: None,
        }
    }

    /// Return the current factor matrices as a tuple `(W, H)`.
    pub fn parameters(&self) -> (Option<&Array2<f64>>, Option<&Array2<f64>>) {
        (self.w.as_ref(), self.h.as_ref())
    }

    fn init_factor_matrices(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>) -> Result<(), String> {
        let (n, m) = x.dim();

        let w_final: Array2<f64>;
        let h_final: Array2<f64>;

        if w.is_none() {
            let mut als = VanillaALS::new(self.k, 0.0, 200, 1e-4);
            als.fit(x, None, None, 1, false)?;
            let als_w = als.w.unwrap();
            let col_norms: Array1<f64> = als_w.axis_iter(Axis(1))
                .map(|col| col.iter().map(|&v| v * v).sum::<f64>().sqrt())
                .collect();
            w_final = Array2::from_shape_fn((n, self.k), |(i, j)| {
                let norm = col_norms[j];
                if norm > 0.0 { als_w[[i, j]] / norm } else { als_w[[i, j]] }
            });
            h_final = if let Some(h) = h {
                h.to_owned()
            } else {
                als.h.unwrap()
            };
        } else if let Some(w) = w {
            w_final = w.to_owned();
            h_final = if let Some(h) = h {
                h.to_owned()
            } else {
                let mut rng = rand::thread_rng();
                Array2::from_shape_fn((self.k, m), |_| rng.r#gen::<f64>().abs())
            };
        } else {
            unreachable!()
        }

        self.w = Some(w_final);
        self.h = Some(h_final);
        Ok(())
    }

    fn loss(&self, x: &Array2<f64>, x_hat: &Array2<f64>) -> f64 {
        (x - x_hat).iter().map(|&v| v * v).sum::<f64>()
    }

    fn update_h(&self, x: &Array2<f64>, w: &Array2<f64>, h: &mut Array2<f64>) {
        let eps = f64::EPSILON;
        let xt_w = x.t().dot(w); // (M, K)
        let wt_w = w.t().dot(w); // (K, K)
        let m = h.ncols();

        for k in 0..self.k {
            let mut update = Array1::<f64>::zeros(m);
            for j in 0..m {
                let ht_dot_wt_w_k: f64 = (0..self.k).map(|kk| h[[kk, j]] * wt_w[[kk, k]]).sum();
                update[j] = xt_w[[j, k]] - ht_dot_wt_w_k;
            }
            for j in 0..m {
                h[[k, j]] = (h[[k, j]] + update[j]).max(eps);
            }
        }
    }

    fn update_w(&self, x: &Array2<f64>, w: &mut Array2<f64>, h: &Array2<f64>) {
        let eps = f64::EPSILON;
        let x_ht = x.dot(&h.t()); // (N, K)
        let hh_t = h.dot(&h.t()); // (K, K)
        let n = w.nrows();

        for k in 0..self.k {
            for i in 0..n {
                let w_dot_hh_t_k: f64 = (0..self.k).map(|kk| w[[i, kk]] * hh_t[[kk, k]]).sum();
                w[[i, k]] = (w[[i, k]] * hh_t[[k, k]] + x_ht[[i, k]] - w_dot_hh_t_k).max(eps);
            }

            let norm = (0..n).map(|i| w[[i, k]] * w[[i, k]]).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..n {
                    w[[i, k]] /= norm;
                }
            }
        }
    }

    /// Factor `x` into nonnegative matrices `W` and `H` via fast HALS.
    pub fn fit(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>, n_initializations: usize, verbose: bool) -> Result<(), String> {
        let n_init = if w.is_some() && h.is_some() { 1 } else { n_initializations };

        let mut best_loss = f64::INFINITY;
        let mut best_w: Option<Array2<f64>> = None;
        let mut best_h: Option<Array2<f64>> = None;

        for f in 0..n_init {
            if verbose {
                println!("\nINITIALIZATION {}", f + 1);
            }

            let (new_w, new_h, loss) = self.fit_once(x, w, h, verbose)?;

            if loss <= best_loss {
                best_loss = loss;
                best_w = Some(new_w);
                best_h = Some(new_h);
            }
        }

        self.w = best_w;
        self.h = best_h;

        if verbose {
            println!("\nFINAL LOSS: {}", best_loss);
        }

        Ok(())
    }

    fn fit_once(&mut self, x: &Array2<f64>, w: Option<&Array2<f64>>, h: Option<&Array2<f64>>, verbose: bool) -> Result<(Array2<f64>, Array2<f64>, f64), String> {
        self.init_factor_matrices(x, w, h)?;
        let mut w = self.w.clone().unwrap();
        let mut h = self.h.clone().unwrap();

        for i in 0..self.max_iter {
            self.update_h(x, &w, &mut h);
            self.update_w(x, &mut w, &h);
            let loss = self.loss(x, &w.dot(&h));

            if verbose {
                println!("[Iter {}] Loss: {:.8}", i + 1, loss);
            }

            if loss <= self.tol {
                break;
            }
        }

        let final_loss = self.loss(x, &w.dot(&h));
        Ok((w, h, final_loss))
    }
}

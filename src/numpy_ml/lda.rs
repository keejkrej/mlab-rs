use ndarray::{Array1, Array2};
use rand::distributions::Distribution;
use special::Gamma;

/// Vanilla (non-smoothed) Latent Dirichlet Allocation trained with variational EM.
pub struct LDA {
    pub n_topics: usize,
    pub alpha: Option<Array1<f64>>,
    pub beta: Option<Array2<f64>>,
    pub phi: Vec<Array2<f64>>,
    pub gamma: Option<Array2<f64>>,
    pub corpus: Vec<Vec<usize>>,
    pub n_docs: usize,
    pub n_words_per_doc: Vec<usize>,
    pub vocab_size: usize,
}

impl LDA {
    /// Create a new LDA model with `n_topics` topics.
    pub fn new(n_topics: usize) -> Self {
        Self {
            n_topics,
            alpha: None,
            beta: None,
            phi: Vec::new(),
            gamma: None,
            corpus: Vec::new(),
            n_docs: 0,
            n_words_per_doc: Vec::new(),
            vocab_size: 0,
        }
    }

    fn maximize_phi(&mut self) {
        let d = self.n_docs;
        let t = self.n_topics;
        let beta = self.beta.as_ref().unwrap();
        let gamma = self.gamma.as_ref().unwrap();
        let corpus = &self.corpus;

        for doc in 0..d {
            let n = self.n_words_per_doc[doc];
            for word in 0..n {
                let w_n = corpus[doc][word];
                let mut sum = 0.0;
                for topic in 0..t {
                    let val = beta[[w_n, topic]] * dg(gamma, doc, topic).exp();
                    self.phi[doc][[word, topic]] = val;
                    sum += val;
                }
                for topic in 0..t {
                    self.phi[doc][[word, topic]] /= sum;
                }
            }
        }
    }

    fn maximize_gamma(&mut self) {
        let d = self.n_docs;
        let t = self.n_topics;
        let alpha = self.alpha.as_ref().unwrap();
        let phi = &self.phi;

        let mut gamma = Array2::<f64>::zeros((d, t));
        for doc in 0..d {
            for topic in 0..t {
                let mut sum = 0.0;
                for word in 0..self.n_words_per_doc[doc] {
                    sum += phi[doc][[word, topic]];
                }
                gamma[[doc, topic]] = alpha[topic] + sum;
            }
        }
        self.gamma = Some(gamma);
    }

    fn maximize_beta(&mut self) {
        let t = self.n_topics;
        let v = self.vocab_size;
        let phi = &self.phi;
        let corpus = &self.corpus;

        let mut beta = Array2::<f64>::zeros((v, t));
        for word in 0..v {
            for topic in 0..t {
                let mut sum = 0.0;
                for doc in 0..self.n_docs {
                    for n in 0..self.n_words_per_doc[doc] {
                        if corpus[doc][n] == word {
                            sum += phi[doc][[n, topic]];
                        }
                    }
                }
                beta[[word, topic]] = sum;
            }
        }

        // Normalize columns
        for topic in 0..t {
            let col_sum: f64 = (0..v).map(|word| beta[[word, topic]]).sum();
            for word in 0..v {
                beta[[word, topic]] /= col_sum;
            }
        }

        self.beta = Some(beta);
    }

    fn maximize_alpha(&mut self, max_iters: usize, tol: f64) {
        let d = self.n_docs;
        let t = self.n_topics;
        let gamma = self.gamma.as_ref().unwrap();

        let mut alpha = self.alpha.as_ref().unwrap().clone();
        for _ in 0..max_iters {
            let alpha_old = alpha.clone();

            let gamma_row_sum: Array1<f64> = gamma.sum_axis(ndarray::Axis(1));

            let mut g = Array1::<f64>::zeros(t);
            for topic in 0..t {
                let digamma_sum_alpha = alpha.sum().digamma();
                let mut sum_digamma_gamma = 0.0;
                for doc in 0..d {
                    sum_digamma_gamma += gamma[[doc, topic]].digamma() - gamma_row_sum[doc].digamma();
                }
                g[topic] = d as f64 * (digamma_sum_alpha - alpha[topic].digamma()) + sum_digamma_gamma;
            }

            let mut h = Array1::<f64>::zeros(t);
            for topic in 0..t {
                h[topic] = -(d as f64) * alpha[topic].trigamma();
            }

            let z = d as f64 * alpha.sum().trigamma();

            let mut c_num = 0.0;
            let mut c_den = 0.0;
            for topic in 0..t {
                c_num += g[topic] / h[topic];
                c_den += 1.0 / h[topic];
            }
            let c = c_num / (z.powi(-1) + c_den);

            for topic in 0..t {
                alpha[topic] -= (g[topic] - c) / h[topic];
            }

            let diff: f64 = alpha.iter().zip(alpha_old.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / t as f64;
            if diff.sqrt() < tol {
                break;
            }
        }

        self.alpha = Some(alpha);
    }

    fn e_step(&mut self) {
        self.maximize_phi();
        self.maximize_gamma();
    }

    fn m_step(&mut self) {
        self.maximize_beta();
        self.maximize_alpha(1000, 0.1);
    }

    /// Compute the variational lower bound for the current parameters.
    pub fn vlb(&self) -> f64 {
        let phi = &self.phi;
        let alpha = self.alpha.as_ref().unwrap();
        let beta = self.beta.as_ref().unwrap();
        let gamma = self.gamma.as_ref().unwrap();
        let corpus = &self.corpus;

        let d = self.n_docs;
        let t = self.n_topics;

        let alpha_sum = alpha.sum();
        let alpha_sum_ln_gamma = special::Gamma::ln_gamma(alpha_sum).0;

        let mut a_total = 0.0;
        let mut d_total = 0.0;
        let mut b_total = 0.0;
        let mut c_total = 0.0;

        for doc in 0..d {
            let alpha_term = alpha_sum_ln_gamma - alpha.iter().map(|&a| special::Gamma::ln_gamma(a).0).sum::<f64>();
            let mut alpha_gamma_term = 0.0;
            for topic in 0..t {
                alpha_gamma_term += (alpha[topic] - 1.0) * dg(gamma, doc, topic);
            }
            a_total += alpha_term + alpha_gamma_term;

            let gamma_sum = gamma.row(doc).sum();
            let gamma_term = special::Gamma::ln_gamma(gamma_sum).0 - gamma.row(doc).iter().map(|&g| special::Gamma::ln_gamma(g).0).sum::<f64>();
            let mut gamma_phi_term = 0.0;
            for topic in 0..t {
                gamma_phi_term += (gamma[[doc, topic]] - 1.0) * dg(gamma, doc, topic);
            }
            d_total += gamma_term + gamma_phi_term;

            for n in 0..self.n_words_per_doc[doc] {
                let w_n = corpus[doc][n];
                for topic in 0..t {
                    let phi_val = phi[doc][[n, topic]];
                    b_total += phi_val * dg(gamma, doc, topic);
                    c_total += phi_val * beta[[w_n, topic]].ln();
                    d_total += phi_val * phi_val.ln();
                }
            }
        }

        a_total + b_total + c_total - d_total
    }

    fn initialize_parameters(&mut self) {
        let t = self.n_topics;
        let v = self.vocab_size;
        let d = self.n_docs;

        let mut rng = rand::thread_rng();

        // alpha ~ 100 * Dirichlet(10 * 1_T)
        let dirichlet = rand_distr::Dirichlet::new_with_size(10.0, t).unwrap();
        self.alpha = Some(Array1::from_vec(dirichlet.sample(&mut rng).iter().map(|&x| 100.0 * x).collect()));

        // beta ~ Dirichlet(1_V) for each topic, transposed -> (V, T)
        let mut beta = Array2::<f64>::zeros((v, t));
        let dirichlet_v = rand_distr::Dirichlet::new_with_size(1.0, v).unwrap();
        for topic in 0..t {
            let sample = dirichlet_v.sample(&mut rng);
            for word in 0..v {
                beta[[word, topic]] = sample[word];
            }
        }
        self.beta = Some(beta);

        // phi uniform
        self.phi = Vec::with_capacity(d);
        for doc in 0..d {
            let n = self.n_words_per_doc[doc];
            self.phi.push(Array2::from_elem((n, t), 1.0 / t as f64));
        }

        // gamma = alpha + N_d / T
        let alpha = self.alpha.as_ref().unwrap();
        let mut gamma = Array2::<f64>::zeros((d, t));
        for doc in 0..d {
            for topic in 0..t {
                gamma[[doc, topic]] = alpha[topic] + self.n_words_per_doc[doc] as f64 / t as f64;
            }
        }
        self.gamma = Some(gamma);
    }

    /// Train the LDA model on a corpus of tokenized documents.
    pub fn train(&mut self, corpus: Vec<Vec<usize>>, verbose: bool, max_iter: usize, tol: f64) {
        self.n_docs = corpus.len();
        self.vocab_size = corpus.iter().flat_map(|doc| doc.iter()).copied().max().map(|m| m + 1).unwrap_or(0);
        self.n_words_per_doc = corpus.iter().map(|doc| doc.len()).collect();
        self.corpus = corpus;

        self.initialize_parameters();
        let mut vlb = f64::NEG_INFINITY;

        for i in 0..max_iter {
            let old_vlb = vlb;

            self.e_step();
            self.m_step();

            vlb = self.vlb();
            let delta = vlb - old_vlb;

            if verbose {
                println!("Iteration {}: {:.3} (delta: {:.2})", i + 1, vlb, delta);
            }

            if delta < tol {
                break;
            }
        }
    }
}

fn dg(gamma: &Array2<f64>, d: usize, t: usize) -> f64 {
    gamma[[d, t]].digamma() - gamma.row(d).sum().digamma()
}

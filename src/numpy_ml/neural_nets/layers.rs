use ndarray::{s, Array2};

use super::activations::{create_activation, Activation};
use super::initializers;
use super::optimizers::{create_optimizer, Optimizer};

/// A fully connected (dense/linear) layer.
///
/// Forward: Y = act_fn(X @ W + b)
pub struct Linear {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    pub parameters: Option<LinearParams>,
    pub gradients: Option<LinearGrads>,
    act_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    init: String,
    x_cache: Vec<Array2<f64>>,
    z_cache: Vec<Array2<f64>>,
}

pub struct LinearParams {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
}

pub struct LinearGrads {
    pub dw: Array2<f64>,
    pub db: Array2<f64>,
}

impl Linear {
    pub fn new(n_out: usize, act_fn: Option<&str>, init: &str, optimizer: Option<&str>) -> Self {
        Linear {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Identity")),
            optimizer: create_optimizer(optimizer.unwrap_or("SGD")),
            init: init.to_string(),
            x_cache: Vec::new(),
            z_cache: Vec::new(),
        }
    }

    fn init_params(&mut self, n_in: usize) {
        self.n_in = n_in;
        let gain = initializers::calc_glorot_gain(self.act_fn.name());
        let w = initializers::glorot_uniform(n_in, self.n_out, gain);
        let b = Array2::zeros((1, self.n_out));
        self.parameters = Some(LinearParams { w, b: b.clone() });
        self.gradients = Some(LinearGrads {
            dw: Array2::zeros((n_in, self.n_out)),
            db: Array2::zeros((1, self.n_out)),
        });
    }

    pub fn forward(&mut self, x: &Array2<f64>, retain_derived: bool) -> Array2<f64> {
        if self.parameters.is_none() {
            self.init_params(x.ncols());
        }

        let params = self.parameters.as_ref().unwrap();
        let z = x.dot(&params.w) + &params.b;
        let y = self.act_fn.fn_(&z);

        if retain_derived {
            self.x_cache.push(x.clone());
            self.z_cache.push(z);
        }

        y
    }

    pub fn backward(&mut self, dldy: &Array2<f64>, retain_grads: bool) -> Array2<f64> {
        assert!(self.trainable, "Layer is frozen");

        let x = self.x_cache.pop().unwrap();
        let z = self.z_cache.pop().unwrap();
        let params = self.parameters.as_ref().unwrap();

        let dz = dldy * &self.act_fn.grad(&z);
        let dw = x.t().dot(&dz);
        let db = dz.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let dx = dz.dot(&params.w.t());

        if retain_grads {
            let grads = self.gradients.as_mut().unwrap();
            grads.dw += &dw;
            grads.db += &db;
        }

        dx
    }

    pub fn update_params(&mut self, cur_loss: Option<f64>) {
        if let (Some(params), Some(grads)) =
            (&self.parameters, &self.gradients)
        {
            let w = self.optimizer.update(&params.w, &grads.dw, "w", cur_loss);
            let b = self.optimizer.update(&params.b, &grads.db, "b", cur_loss);
            self.parameters = Some(LinearParams { w, b });
        }
    }

    pub fn flush_gradients(&mut self) {
        if let Some(ref params) = self.parameters {
            if let Some(ref mut grads) = self.gradients {
                grads.dw = Array2::zeros(params.w.raw_dim());
                grads.db = Array2::zeros(params.b.raw_dim());
            }
        }
        self.x_cache.clear();
        self.z_cache.clear();
    }

    pub fn freeze(&mut self) {
        self.trainable = false;
    }

    pub fn unfreeze(&mut self) {
        self.trainable = true;
    }
}

/// A vanilla (Elman) RNN cell.
///
/// A[t] = act_fn(Wax @ Xt + Waa @ A[t-1] + ba)
pub struct RNNCell {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    parameters: Option<RNNParams>,
    gradients: Option<RNNGrads>,
    act_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    a_cache: Vec<Array2<f64>>,
    z_cache: Vec<Array2<f64>>,
    x_cache: Vec<Array2<f64>>,
    current_step: usize,
    dldx_acc: Option<Array2<f64>>,
}

struct RNNParams {
    wax: Array2<f64>,
    waa: Array2<f64>,
    ba: Array2<f64>,
}

struct RNNGrads {
    wax: Array2<f64>,
    waa: Array2<f64>,
    ba: Array2<f64>,
}

impl RNNCell {
    pub fn new(n_out: usize, act_fn: Option<&str>, _init: &str, _optimizer: Option<&str>) -> Self {
        RNNCell {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Tanh")),
            optimizer: create_optimizer(_optimizer.unwrap_or("SGD")),
            a_cache: Vec::new(),
            z_cache: Vec::new(),
            x_cache: Vec::new(),
            current_step: 0,
            dldx_acc: None,
        }
    }

    fn init_params(&mut self) {
        let gain = initializers::calc_glorot_gain(self.act_fn.name());
        let wax = initializers::glorot_uniform(self.n_in, self.n_out, gain);
        let waa = initializers::glorot_uniform(self.n_out, self.n_out, gain);
        let ba = Array2::zeros((1, self.n_out));

        self.parameters = Some(RNNParams {
            wax,
            waa,
            ba: ba.clone(),
        });
        self.gradients = Some(RNNGrads {
            wax: Array2::zeros((self.n_in, self.n_out)),
            waa: Array2::zeros((self.n_out, self.n_out)),
            ba,
        });
    }

    pub fn forward(&mut self, xt: &Array2<f64>) -> Array2<f64> {
        if self.parameters.is_none() {
            self.n_in = xt.ncols();
            self.init_params();
        }

        let params = self.parameters.as_ref().unwrap();
        let a_prev = if self.a_cache.is_empty() {
            Array2::zeros((xt.nrows(), self.n_out))
        } else {
            self.a_cache.last().unwrap().clone()
        };

        let z = xt.dot(&params.wax) + a_prev.dot(&params.waa) + &params.ba;
        let at = self.act_fn.fn_(&z);

        self.z_cache.push(z);
        self.a_cache.push(at.clone());
        self.x_cache.push(xt.clone());

        at
    }

    pub fn backward(&mut self, dldat: &Array2<f64>) -> Array2<f64> {
        assert!(self.trainable, "Layer is frozen");

        if self.current_step == 0 {
            self.current_step = self.a_cache.len() - 1;
        }

        let t = self.current_step;
        let params = self.parameters.as_ref().unwrap();

        let z_t = &self.z_cache[t];
        let a_prev = if t > 0 { &self.a_cache[t - 1] } else { &self.a_cache[0] };
        let x_t = &self.x_cache[t];

        let dldx_acc = self.dldx_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });

        let da = dldat + &dldx_acc;
        let dz = &self.act_fn.grad(z_t) * &da;

        let dwax = x_t.t().dot(&dz);
        let dwaa = a_prev.t().dot(&dz);
        let dba = dz.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let dx = dz.dot(&params.wax.t());
        let da_prev = dz.dot(&params.waa.t());

        self.dldx_acc = Some(da_prev);
        self.current_step = t.wrapping_sub(1);

        if let Some(ref mut grads) = self.gradients {
            grads.wax += &dwax;
            grads.waa += &dwaa;
            grads.ba += &dba;
        }

        dx
    }

    pub fn flush_gradients(&mut self) {
        if let Some(ref params) = self.parameters {
            if let Some(ref mut grads) = self.gradients {
                grads.wax = Array2::zeros(params.wax.raw_dim());
                grads.waa = Array2::zeros(params.waa.raw_dim());
                grads.ba = Array2::zeros(params.ba.raw_dim());
            }
        }
        self.a_cache.clear();
        self.z_cache.clear();
        self.x_cache.clear();
        self.current_step = 0;
        self.dldx_acc = None;
    }
}

/// A single LSTM cell.
///
/// Z[t] = [A[t-1], X[t]]
/// Gf[t] = gate_fn(Wf @ Z[t] + bf)  (forget gate)
/// Gu[t] = gate_fn(Wu @ Z[t] + bu)  (update gate)
/// Go[t] = gate_fn(Wo @ Z[t] + bo)  (output gate)
/// Cc[t] = act_fn(Wc @ Z[t] + bc)   (candidate)
/// C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
/// A[t]  = Go[t] * act_fn(C[t])
pub struct LSTMCell {
    pub n_in: usize,
    pub n_out: usize,
    pub trainable: bool,
    parameters: Option<LSTMParams>,
    gradients: Option<LSTMGrads>,
    act_fn: Box<dyn Activation>,
    gate_fn: Box<dyn Activation>,
    optimizer: Box<dyn Optimizer>,
    a_cache: Vec<Array2<f64>>,
    c_cache: Vec<Array2<f64>>,
    gf_cache: Vec<Array2<f64>>,
    gu_cache: Vec<Array2<f64>>,
    go_cache: Vec<Array2<f64>>,
    cc_cache: Vec<Array2<f64>>,
    x_cache: Vec<Array2<f64>>,
    current_step: usize,
    dlda_acc: Option<Array2<f64>>,
    dldc_acc: Option<Array2<f64>>,
}

struct LSTMParams {
    wf: Array2<f64>,
    wu: Array2<f64>,
    wc: Array2<f64>,
    wo: Array2<f64>,
    bf: Array2<f64>,
    bu: Array2<f64>,
    bc: Array2<f64>,
    bo: Array2<f64>,
}

struct LSTMGrads {
    wf: Array2<f64>,
    wu: Array2<f64>,
    wc: Array2<f64>,
    wo: Array2<f64>,
    bf: Array2<f64>,
    bu: Array2<f64>,
    bc: Array2<f64>,
    bo: Array2<f64>,
}

impl LSTMCell {
    pub fn new(
        n_out: usize,
        act_fn: Option<&str>,
        gate_fn: Option<&str>,
        _init: &str,
        _optimizer: Option<&str>,
    ) -> Self {
        LSTMCell {
            n_in: 0,
            n_out,
            trainable: true,
            parameters: None,
            gradients: None,
            act_fn: create_activation(act_fn.unwrap_or("Tanh")),
            gate_fn: create_activation(gate_fn.unwrap_or("Sigmoid")),
            optimizer: create_optimizer(_optimizer.unwrap_or("SGD")),
            a_cache: Vec::new(),
            c_cache: Vec::new(),
            gf_cache: Vec::new(),
            gu_cache: Vec::new(),
            go_cache: Vec::new(),
            cc_cache: Vec::new(),
            x_cache: Vec::new(),
            current_step: 0,
            dlda_acc: None,
            dldc_acc: None,
        }
    }

    fn init_params(&mut self) {
        let input_dim = self.n_in + self.n_out;
        let gain = initializers::calc_glorot_gain(self.gate_fn.name());
        let gain_act = initializers::calc_glorot_gain(self.act_fn.name());

        let wf = initializers::glorot_uniform(input_dim, self.n_out, gain);
        let wu = initializers::glorot_uniform(input_dim, self.n_out, gain);
        let wc = initializers::glorot_uniform(input_dim, self.n_out, gain_act);
        let wo = initializers::glorot_uniform(input_dim, self.n_out, gain);

        let bf = Array2::zeros((1, self.n_out));
        let bu = Array2::zeros((1, self.n_out));
        let bc = Array2::zeros((1, self.n_out));
        let bo = Array2::zeros((1, self.n_out));

        self.parameters = Some(LSTMParams {
            wf: wf.clone(),
            wu: wu.clone(),
            wc: wc.clone(),
            wo: wo.clone(),
            bf: bf.clone(),
            bu: bu.clone(),
            bc: bc.clone(),
            bo: bo.clone(),
        });

        self.gradients = Some(LSTMGrads {
            wf: Array2::zeros(wf.raw_dim()),
            wu: Array2::zeros(wu.raw_dim()),
            wc: Array2::zeros(wc.raw_dim()),
            wo: Array2::zeros(wo.raw_dim()),
            bf: Array2::zeros(bf.raw_dim()),
            bu: Array2::zeros(bu.raw_dim()),
            bc: Array2::zeros(bc.raw_dim()),
            bo: Array2::zeros(bo.raw_dim()),
        });
    }

    pub fn forward(&mut self, xt: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        if self.parameters.is_none() {
            self.n_in = xt.ncols();
            self.init_params();
        }

        let params = self.parameters.as_ref().unwrap();

        let (a_prev, c_prev) = if self.a_cache.is_empty() {
            let zeros = Array2::zeros((xt.nrows(), self.n_out));
            (zeros.clone(), zeros)
        } else {
            (
                self.a_cache.last().unwrap().clone(),
                self.c_cache.last().unwrap().clone(),
            )
        };

        // Concatenate [A_prev, X_t] by horizontal stacking
        let (n_ex, _) = a_prev.dim();
        let mut z = Array2::zeros((n_ex, self.n_in + self.n_out));
        for i in 0..n_ex {
            for j in 0..self.n_out {
                z[[i, j]] = a_prev[[i, j]];
            }
            for j in 0..self.n_in {
                z[[i, self.n_out + j]] = xt[[i, j]];
            }
        }

        let gf = self.gate_fn.fn_(&(z.dot(&params.wf) + &params.bf));
        let gu = self.gate_fn.fn_(&(z.dot(&params.wu) + &params.bu));
        let go = self.gate_fn.fn_(&(z.dot(&params.wo) + &params.bo));
        let cc = self.act_fn.fn_(&(z.dot(&params.wc) + &params.bc));
        let c = &gf * &c_prev + &gu * &cc;
        let a = &go * &self.act_fn.fn_(&c);

        self.x_cache.push(xt.clone());
        self.a_cache.push(a.clone());
        self.c_cache.push(c.clone());
        self.gf_cache.push(gf);
        self.gu_cache.push(gu);
        self.go_cache.push(go);
        self.cc_cache.push(cc);

        (a, c)
    }

    pub fn backward(&mut self, dldat: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        assert!(self.trainable, "Layer is frozen");

        if self.current_step == 0 {
            self.current_step = self.a_cache.len() - 1;
        }

        let t = self.current_step;
        let params = self.parameters.as_ref().unwrap();

        let ct = &self.c_cache[t + 1];
        let c_prev = &self.c_cache[t];
        let zt = {
            let a_prev = if t > 0 { &self.a_cache[t - 1] } else { &self.a_cache[0] };
            let xt = &self.x_cache[t];
            let (n_ex, _) = a_prev.dim();
            let mut z = Array2::zeros((n_ex, self.n_in + self.n_out));
            for i in 0..n_ex {
                for j in 0..self.n_out {
                    z[[i, j]] = a_prev[[i, j]];
                }
                for j in 0..self.n_in {
                    z[[i, self.n_out + j]] = xt[[i, j]];
                }
            }
            z
        };

        let dlda_acc = self.dlda_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });
        let dldc_acc = self.dldc_acc.take().unwrap_or_else(|| {
            Array2::zeros((dldat.nrows(), self.n_out))
        });

        let da = dldat + &dlda_acc;
        let dc = &dldc_acc + &da * &self.go_cache[t] * &self.act_fn.grad(ct);

        let _go = zt.dot(&params.wo) + &params.bo;
        let _gf = zt.dot(&params.wf) + &params.bf;
        let _gu = zt.dot(&params.wu) + &params.bu;
        let _gc = zt.dot(&params.wc) + &params.bc;

        let dgo = &da * &self.act_fn.fn_(ct) * &self.gate_fn.grad(&_go);
        let dgc = &dc * &self.gu_cache[t] * &self.act_fn.grad(&_gc);
        let dgu = &dc * &self.cc_cache[t] * &self.gate_fn.grad(&_gu);
        let dgf = &dc * c_prev * &self.gate_fn.grad(&_gf);

        let dz = dgf.dot(&params.wf.t())
            + dgu.dot(&params.wu.t())
            + dgc.dot(&params.wc.t())
            + dgo.dot(&params.wo.t());

        let n_out = self.n_out;
        let dx = dz.slice(s![.., n_out..]).to_owned();
        let da_prev = dz.slice(s![.., ..n_out]).to_owned();

        // Update parameter gradients
        if let Some(ref mut grads) = self.gradients {
            grads.wf += &zt.t().dot(&dgf);
            grads.wu += &zt.t().dot(&dgu);
            grads.wc += &zt.t().dot(&dgc);
            grads.wo += &zt.t().dot(&dgo);
            grads.bf += &dgf.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bu += &dgu.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bc += &dgc.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
            grads.bo += &dgo.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        }

        self.dlda_acc = Some(da_prev);
        self.dldc_acc = Some(&self.gf_cache[t] * &dc);
        self.current_step = t.wrapping_sub(1);

        (dx, da)
    }

    pub fn flush_gradients(&mut self) {
        self.a_cache.clear();
        self.c_cache.clear();
        self.gf_cache.clear();
        self.gu_cache.clear();
        self.go_cache.clear();
        self.cc_cache.clear();
        self.x_cache.clear();
        self.current_step = 0;
        self.dlda_acc = None;
        self.dldc_acc = None;
    }
}

/// A full RNN layer that processes a sequence.
pub struct RNNLayer {
    pub n_out: usize,
    cell: RNNCell,
}

impl RNNLayer {
    pub fn new(n_out: usize, act_fn: Option<&str>, init: &str, optimizer: Option<&str>) -> Self {
        RNNLayer {
            n_out,
            cell: RNNCell::new(n_out, act_fn, init, optimizer),
        }
    }

    /// Forward pass over a full sequence.
    /// X shape: (n_ex * n_t, n_in) - flattened sequence
    pub fn forward(&mut self, x: &Array2<f64>, n_t: usize) -> Vec<Array2<f64>> {
        let mut outputs = Vec::with_capacity(n_t);
        let n_ex = x.nrows() / n_t;
        for t in 0..n_t {
            let xt = x.slice(s![t * n_ex..(t + 1) * n_ex, ..]).to_owned();
            let yt = self.cell.forward(&xt);
            outputs.push(yt);
        }
        outputs
    }

    /// Backward pass over a full sequence.
    pub fn backward(&mut self, dldy: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut dxs = Vec::new();
        for t in (0..dldy.len()).rev() {
            let dxt = self.cell.backward(&dldy[t]);
            dxs.insert(0, dxt);
        }
        dxs
    }
}

/// A full LSTM layer that processes a sequence.
pub struct LSTMLayer {
    pub n_out: usize,
    cell: LSTMCell,
}

impl LSTMLayer {
    pub fn new(
        n_out: usize,
        act_fn: Option<&str>,
        gate_fn: Option<&str>,
        init: &str,
        optimizer: Option<&str>,
    ) -> Self {
        LSTMLayer {
            n_out,
            cell: LSTMCell::new(n_out, act_fn, gate_fn, init, optimizer),
        }
    }

    /// Forward pass over a full sequence.
    pub fn forward(&mut self, x: &Array2<f64>, n_t: usize) -> Vec<Array2<f64>> {
        let mut outputs = Vec::with_capacity(n_t);
        let n_ex = x.nrows() / n_t;
        for t in 0..n_t {
            let xt = x.slice(s![t * n_ex..(t + 1) * n_ex, ..]).to_owned();
            let (at, _ct) = self.cell.forward(&xt);
            outputs.push(at);
        }
        outputs
    }

    /// Backward pass over a full sequence.
    pub fn backward(&mut self, dldy: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut dxs = Vec::new();
        for t in (0..dldy.len()).rev() {
            let (dxt, _da_prev) = self.cell.backward(&dldy[t]);
            dxs.insert(0, dxt);
        }
        dxs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear::new(3, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = layer.forward(&x, false);
        assert_eq!(y.dim(), (2, 3));
    }

    #[test]
    fn test_linear_backward() {
        let mut layer = Linear::new(3, Some("ReLU"), "glorot_uniform", Some("SGD"));
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = layer.forward(&x, true);
        let dldy = Array2::ones(y.raw_dim());
        let dx = layer.backward(&dldy, true);
        assert_eq!(dx.dim(), x.dim());
    }

    #[test]
    fn test_rnn_cell_forward() {
        let mut cell = RNNCell::new(4, Some("Tanh"), "glorot_uniform", Some("SGD"));
        let xt = array![[1.0, 2.0, 3.0]];
        let at = cell.forward(&xt);
        assert_eq!(at.dim(), (1, 4));
    }

    #[test]
    fn test_lstm_cell_forward() {
        let mut cell = LSTMCell::new(4, Some("Tanh"), Some("Sigmoid"), "glorot_uniform", Some("SGD"));
        let xt = array![[1.0, 2.0, 3.0]];
        let (at, ct) = cell.forward(&xt);
        assert_eq!(at.dim(), (1, 4));
        assert_eq!(ct.dim(), (1, 4));
    }
}

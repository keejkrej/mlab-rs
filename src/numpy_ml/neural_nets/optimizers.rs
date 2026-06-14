use std::collections::HashMap;

use ndarray::Array2;

use super::schedulers::Scheduler;

/// Trait for all optimizers.
pub trait Optimizer {
    /// Update a parameter given its gradient.
    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64>;
    /// Increment the step counter.
    fn step(&mut self);
    /// Return the name of the optimizer.
    fn name(&self) -> &str;
}

/// SGD optimizer with optional momentum and gradient clipping.
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub clip_norm: Option<f64>,
    pub cur_step: usize,
    cache: HashMap<String, Array2<f64>>,
    lr_scheduler: Box<dyn Scheduler>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, clip_norm: Option<f64>) -> Self {
        use super::schedulers::ConstantScheduler;
        SGD {
            lr,
            momentum,
            clip_norm,
            cur_step: 0,
            cache: HashMap::new(),
            lr_scheduler: Box::new(ConstantScheduler::new(lr)),
        }
    }

    pub fn with_scheduler(lr: f64, momentum: f64, clip_norm: Option<f64>, scheduler: Box<dyn Scheduler>) -> Self {
        SGD {
            lr,
            momentum,
            clip_norm,
            cur_step: 0,
            cache: HashMap::new(),
            lr_scheduler: scheduler,
        }
    }
}

impl Optimizer for SGD {
    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64> {
        let lr = self.lr_scheduler.learning_rate(self.cur_step, cur_loss);
        let t = self.clip_norm.unwrap_or(f64::INFINITY);
        let grad_norm = param_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grad = if grad_norm > t {
            param_grad * (t / grad_norm)
        } else {
            param_grad.clone()
        };

        let prev_update = self
            .cache
            .get(param_name)
            .cloned()
            .unwrap_or_else(|| Array2::zeros(param.raw_dim()));

        let update = &prev_update * self.momentum + &grad * lr;
        self.cache.insert(param_name.to_string(), update.clone());

        param - &update
    }

    fn step(&mut self) {
        self.cur_step += 1;
    }

    fn name(&self) -> &str {
        "SGD"
    }
}

/// AdaGrad optimizer.
pub struct AdaGrad {
    pub lr: f64,
    pub eps: f64,
    pub clip_norm: Option<f64>,
    pub cur_step: usize,
    cache: HashMap<String, Array2<f64>>,
    lr_scheduler: Box<dyn Scheduler>,
}

impl AdaGrad {
    pub fn new(lr: f64, eps: f64, clip_norm: Option<f64>) -> Self {
        use super::schedulers::ConstantScheduler;
        AdaGrad {
            lr,
            eps,
            clip_norm,
            cur_step: 0,
            cache: HashMap::new(),
            lr_scheduler: Box::new(ConstantScheduler::new(lr)),
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64> {
        let lr = self.lr_scheduler.learning_rate(self.cur_step, cur_loss);
        let t = self.clip_norm.unwrap_or(f64::INFINITY);
        let grad_norm = param_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grad = if grad_norm > t {
            param_grad * (t / grad_norm)
        } else {
            param_grad.clone()
        };

        let c = self
            .cache
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.raw_dim()));

        *c = c.clone() + &grad.mapv(|v| v * v);
        let update = &grad * lr / (c.mapv(|v| v.sqrt()) + self.eps);
        param - &update
    }

    fn step(&mut self) {
        self.cur_step += 1;
    }

    fn name(&self) -> &str {
        "AdaGrad"
    }
}

/// RMSProp optimizer.
pub struct RMSProp {
    pub lr: f64,
    pub decay: f64,
    pub eps: f64,
    pub clip_norm: Option<f64>,
    pub cur_step: usize,
    cache: HashMap<String, Array2<f64>>,
    lr_scheduler: Box<dyn Scheduler>,
}

impl RMSProp {
    pub fn new(lr: f64, decay: f64, eps: f64, clip_norm: Option<f64>) -> Self {
        use super::schedulers::ConstantScheduler;
        RMSProp {
            lr,
            decay,
            eps,
            clip_norm,
            cur_step: 0,
            cache: HashMap::new(),
            lr_scheduler: Box::new(ConstantScheduler::new(lr)),
        }
    }
}

impl Optimizer for RMSProp {
    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64> {
        let lr = self.lr_scheduler.learning_rate(self.cur_step, cur_loss);
        let t = self.clip_norm.unwrap_or(f64::INFINITY);
        let grad_norm = param_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grad = if grad_norm > t {
            param_grad * (t / grad_norm)
        } else {
            param_grad.clone()
        };

        let c = self
            .cache
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.raw_dim()));

        *c = &*c * self.decay + &grad.mapv(|v| v * v) * (1.0 - self.decay);
        let update = &grad * lr / (c.mapv(|v| v.sqrt()) + self.eps);
        param - &update
    }

    fn step(&mut self) {
        self.cur_step += 1;
    }

    fn name(&self) -> &str {
        "RMSProp"
    }
}

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub decay1: f64,
    pub decay2: f64,
    pub eps: f64,
    pub clip_norm: Option<f64>,
    pub cur_step: usize,
    cache: HashMap<String, AdamCache>,
    lr_scheduler: Box<dyn Scheduler>,
}

struct AdamCache {
    t: usize,
    mean: Array2<f64>,
    var: Array2<f64>,
}

impl Adam {
    pub fn new(lr: f64, decay1: f64, decay2: f64, eps: f64, clip_norm: Option<f64>) -> Self {
        use super::schedulers::ConstantScheduler;
        Adam {
            lr,
            decay1,
            decay2,
            eps,
            clip_norm,
            cur_step: 0,
            cache: HashMap::new(),
            lr_scheduler: Box::new(ConstantScheduler::new(lr)),
        }
    }
}

impl Optimizer for Adam {
    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64> {
        let lr = self.lr_scheduler.learning_rate(self.cur_step, cur_loss);
        let t = self.clip_norm.unwrap_or(f64::INFINITY);
        let grad_norm = param_grad.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grad = if grad_norm > t {
            param_grad * (t / grad_norm)
        } else {
            param_grad.clone()
        };

        let c = self.cache.entry(param_name.to_string()).or_insert_with(|| {
            AdamCache {
                t: 0,
                mean: Array2::zeros(param.raw_dim()),
                var: Array2::zeros(param.raw_dim()),
            }
        });

        c.t += 1;
        c.var = &c.var * self.decay2 + &grad.mapv(|v| v * v) * (1.0 - self.decay2);
        c.mean = &c.mean * self.decay1 + &grad * (1.0 - self.decay1);

        let t = c.t as f64;
        let v_hat = &c.var / (1.0 - self.decay2.powf(t));
        let m_hat = &c.mean / (1.0 - self.decay1.powf(t));

        let update = &m_hat * lr / (v_hat.mapv(|v| v.sqrt()) + self.eps);
        param - &update
    }

    fn step(&mut self) {
        self.cur_step += 1;
    }

    fn name(&self) -> &str {
        "Adam"
    }
}

/// Create an optimizer by name string.
pub fn create_optimizer(name: &str) -> Box<dyn Optimizer> {
    match name {
        "SGD" => Box::new(SGD::new(0.01, 0.0, None)),
        "AdaGrad" => Box::new(AdaGrad::new(0.01, 1e-7, None)),
        "RMSProp" => Box::new(RMSProp::new(0.001, 0.9, 1e-7, None)),
        "Adam" => Box::new(Adam::new(0.001, 0.9, 0.999, 1e-7, None)),
        _ => Box::new(SGD::new(0.01, 0.0, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sgd() {
        let mut opt = SGD::new(0.1, 0.0, None);
        let param = array![[1.0, 2.0]];
        let grad = array![[0.1, 0.2]];
        let new_param = opt.update(&param, &grad, "w", None);
        assert!((new_param[[0, 0]] - 0.99).abs() < 1e-6);
        assert!((new_param[[0, 1]] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn test_adam() {
        let mut opt = Adam::new(0.001, 0.9, 0.999, 1e-7, None);
        let param = array![[1.0, 2.0]];
        let grad = array![[0.1, 0.2]];
        let _ = opt.update(&param, &grad, "w", None);
        // After one step, parameter should have moved slightly
        opt.step();
    }
}

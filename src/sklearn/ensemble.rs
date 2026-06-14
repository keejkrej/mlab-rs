use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Internal regression tree (CART with MSE) used by gradient boosting models
// ---------------------------------------------------------------------------

struct RegNode {
    feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<RegNode>>,
    right: Option<Box<RegNode>>,
    value: Option<f64>,
}

struct RegressionTree {
    max_depth: usize,
    min_samples_split: usize,
    root: Option<Box<RegNode>>,
}

impl RegressionTree {
    fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self { max_depth, min_samples_split, root: None }
    }

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        let mut indices: Vec<usize> = (0..nrows).collect();
        self.root = Some(Box::new(self.build(x, y, &mut indices, 0, ncols)));
    }

    fn build(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &mut [usize],
        depth: usize,
        ncols: usize,
    ) -> RegNode {
        let n = indices.len();
        if n == 0 {
            return RegNode { feature: None, threshold: None, left: None, right: None, value: Some(0.0) };
        }

        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;

        if n == 1 || depth >= self.max_depth || n < self.min_samples_split {
            return RegNode { feature: None, threshold: None, left: None, right: None, value: Some(mean) };
        }

        let mut best_mse = f64::INFINITY;
        let mut best_feat = None;
        let mut best_thresh = None;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for feat in 0..ncols {
            let mut vals: Vec<f64> = indices.iter().map(|&i| x[[i, feat]]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();

            for &thresh in &vals {
                let (mut li, mut ri) = (Vec::new(), Vec::new());
                for &i in indices.iter() {
                    if x[[i, feat]] <= thresh { li.push(i); } else { ri.push(i); }
                }
                if li.is_empty() || ri.is_empty() { continue; }

                let mse_l = mse(y, &li);
                let mse_r = mse(y, &ri);
                let weighted = (li.len() as f64) * mse_l + (ri.len() as f64) * mse_r;
                if weighted < best_mse {
                    best_mse = weighted;
                    best_feat = Some(feat);
                    best_thresh = Some(thresh);
                    best_left = li;
                    best_right = ri;
                }
            }
        }

        if best_feat.is_none() {
            return RegNode { feature: None, threshold: None, left: None, right: None, value: Some(mean) };
        }

        let left = self.build(x, y, &mut best_left, depth + 1, ncols);
        let right = self.build(x, y, &mut best_right, depth + 1, ncols);
        RegNode { feature: best_feat, threshold: best_thresh, left: Some(Box::new(left)), right: Some(Box::new(right)), value: None }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let root = self.root.as_ref().expect("RegressionTree not fitted");
        let mut preds = Array1::zeros(x.nrows());
        for r in 0..x.nrows() {
            preds[r] = self.predict_row(root, x, r);
        }
        preds
    }

    fn predict_row(&self, node: &RegNode, x: &Array2<f64>, r: usize) -> f64 {
        if let Some(v) = node.value { return v; }
        let feat = node.feature.unwrap();
        let thresh = node.threshold.unwrap();
        if x[[r, feat]] <= thresh {
            self.predict_row(node.left.as_ref().unwrap(), x, r)
        } else {
            self.predict_row(node.right.as_ref().unwrap(), x, r)
        }
    }
}

fn mse(y: &Array1<f64>, indices: &[usize]) -> f64 {
    let n = indices.len() as f64;
    if n == 0.0 { return 0.0; }
    let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
}

/// Very small random-forest-style classifier built from bootstrap samples of the DecisionTreeClassifier.
pub struct RandomForestClassifier {
    pub n_estimators: usize,
    trees: Vec<crate::sklearn::tree::DecisionTreeClassifier>,
}

impl RandomForestClassifier {
    pub fn new(n_estimators: usize) -> Self {
        Self { n_estimators, trees: Vec::new() }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.trees.clear();
        let mut rng = rand::thread_rng();
        for _ in 0..self.n_estimators {
            let mut indices: Vec<usize> = (0..x.nrows()).collect();
            indices.shuffle(&mut rng);
            let sample_size = x.nrows().max(1);
            let sample_indices = &indices[..sample_size];
            let mut tree = crate::sklearn::tree::DecisionTreeClassifier::new(4, 2);
            let x_boot = x.select(Axis(0), sample_indices);
            let y_boot = y.select(Axis(0), sample_indices);
            tree.fit(&x_boot, &y_boot);
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut preds = Array1::zeros(x.nrows());
        for r in 0..x.nrows() {
            let mut votes = std::collections::HashMap::new();
            for tree in &self.trees {
                let row = x.slice(ndarray::s![r..r + 1, ..]).to_owned();
                let label = tree.predict(&row).into_iter().next().unwrap_or(0.0);
                *votes.entry(label.to_bits()).or_insert(0usize) += 1;
            }
            let majority = votes.into_iter().max_by_key(|(_, count)| *count).map(|(bits, _)| f64::from_bits(bits)).unwrap_or(0.0);
            preds[r] = majority;
        }
        preds
    }
}

// ---------------------------------------------------------------------------
// GradientBoostingRegressor
// ---------------------------------------------------------------------------

pub struct GradientBoostingRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    trees: Vec<RegressionTree>,
    initial_prediction: f64,
}

impl GradientBoostingRegressor {
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize, min_samples_split: usize) -> Self {
        Self { n_estimators, learning_rate, max_depth, min_samples_split, trees: Vec::new(), initial_prediction: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.trees.clear();
        let n = y.len() as f64;
        self.initial_prediction = y.sum() / n;
        let mut current_pred = Array1::from_elem(y.dim(), self.initial_prediction);

        for _ in 0..self.n_estimators {
            let residuals = y - &current_pred;
            let mut tree = RegressionTree::new(self.max_depth, self.min_samples_split);
            tree.fit(x, &residuals);
            let update = tree.predict(x);
            current_pred = current_pred + self.learning_rate * &update;
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut pred = Array1::from_elem(x.nrows(), self.initial_prediction);
        for tree in &self.trees {
            pred = pred + self.learning_rate * &tree.predict(x);
        }
        pred
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let pred = self.predict(x);
        crate::sklearn::metrics::r2_score(y, &pred)
    }
}

// ---------------------------------------------------------------------------
// GradientBoostingClassifier  (binary, log-loss)
// ---------------------------------------------------------------------------

pub struct GradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    trees: Vec<RegressionTree>,
    initial_f: f64,
}

impl GradientBoostingClassifier {
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self { n_estimators, learning_rate, max_depth, min_samples_split: 2, trees: Vec::new(), initial_f: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.trees.clear();
        let n = y.len() as f64;
        let p = y.sum() / n;
        self.initial_f = (p / (1.0 - p)).ln();
        let mut f = Array1::from_elem(y.dim(), self.initial_f);

        for _ in 0..self.n_estimators {
            let proba = f.mapv(sigmoid);
            let residuals = y - &proba;
            let mut tree = RegressionTree::new(self.max_depth, self.min_samples_split);
            tree.fit(x, &residuals);
            let update = tree.predict(x);
            f = f + self.learning_rate * &update;
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.predict_proba(x).column(1).mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 })
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut f = Array1::from_elem(x.nrows(), self.initial_f);
        for tree in &self.trees {
            f = f + self.learning_rate * &tree.predict(x);
        }
        let mut proba = Array2::zeros((x.nrows(), 2));
        for i in 0..x.nrows() {
            let p1 = sigmoid(f[i]);
            proba[[i, 0]] = 1.0 - p1;
            proba[[i, 1]] = p1;
        }
        proba
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_classifier() {
        let clf = RandomForestClassifier::new(3);
        let preds = clf.predict(&Array2::zeros((2, 2)));
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_gradient_boosting_regressor() {
        // y ≈ 2*x + noise
        let n = 100;
        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let xi = i as f64 / n as f64;
            x[[i, 0]] = xi;
            y[i] = 2.0 * xi + 0.05 * rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let mut reg = GradientBoostingRegressor::new(10, 0.1, 3, 2);
        reg.fit(&x, &y);
        let preds = reg.predict(&x);
        let r2 = reg.score(&x, &y);
        assert!(r2 > 0.8, "R² should be > 0.8, got {}", r2);
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_gradient_boosting_classifier() {
        // XOR-like data: class 1 if x0*x1 > 0 (both positive or both negative)
        let n = 200;
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let a: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            let b: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            x[[i, 0]] = a;
            x[[i, 1]] = b;
            y[i] = if a * b > 0.0 { 1.0 } else { 0.0 };
        }
        let mut clf = GradientBoostingClassifier::new(100, 0.1, 3);
        clf.fit(&x, &y);
        let preds = clf.predict(&x);
        let acc = crate::sklearn::metrics::accuracy_score(&y, &preds);
        assert!(acc > 0.9, "accuracy should be > 0.9, got {}", acc);

        let proba = clf.predict_proba(&x);
        assert_eq!(proba.dim(), (n, 2));
        for i in 0..n {
            assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }
}

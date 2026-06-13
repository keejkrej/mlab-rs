use ndarray::{Array1, Array2};

/// A Decision Tree classifier based on the CART algorithm using Gini impurity.
pub struct DecisionTreeClassifier {
    pub max_depth: usize,
    pub min_samples_split: usize,
    root: Option<Box<Node>>,
}

struct Node {
    feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<f64>, // Class label (leaf value)
}

impl DecisionTreeClassifier {
    /// Create a new Decision Tree classifier with maximum depth and minimum samples to split a node.
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
        }
    }

    /// Build the decision tree from training dataset X and labels y.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let (nrows, ncols) = x.dim();
        assert_eq!(nrows, y.len(), "X and y must have same number of samples");

        let mut sample_indices: Vec<usize> = (0..nrows).collect();
        self.root = Some(Box::new(self.build_tree(x, y, &mut sample_indices, 0, ncols)));
    }

    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_indices: &mut [usize],
        depth: usize,
        ncols: usize,
    ) -> Node {
        let n_samples = sample_indices.len();

        if n_samples == 0 {
            return Node {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(0.0),
            };
        }

        let mut counts = std::collections::HashMap::new();
        for &idx in sample_indices.iter() {
            let val = y[idx];
            *counts.entry(val.to_bits()).or_insert(0) += 1;
        }

        let is_pure = counts.len() <= 1;
        let majority_class = counts
            .iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(&bits, _)| f64::from_bits(bits))
            .unwrap_or(0.0);

        if is_pure || depth >= self.max_depth || n_samples < self.min_samples_split {
            return Node {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(majority_class),
            };
        }

        let mut best_gini = 1.0;
        let mut best_feature = None;
        let mut best_threshold = None;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        for feat in 0..ncols {
            let mut feat_values: Vec<f64> = sample_indices.iter().map(|&idx| x[[idx, feat]]).collect();
            feat_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            feat_values.dedup();

            for &thresh in feat_values.iter() {
                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                for &idx in sample_indices.iter() {
                    if x[[idx, feat]] <= thresh {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let gini_left = self.gini_impurity(y, &left_indices);
                let gini_right = self.gini_impurity(y, &right_indices);

                let weight_left = (left_indices.len() as f64) / (n_samples as f64);
                let weight_right = (right_indices.len() as f64) / (n_samples as f64);
                let gini = weight_left * gini_left + weight_right * gini_right;

                if gini < best_gini {
                    best_gini = gini;
                    best_feature = Some(feat);
                    best_threshold = Some(thresh);
                    best_left_indices = left_indices;
                    best_right_indices = right_indices;
                }
            }
        }

        if best_feature.is_none() {
            return Node {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(majority_class),
            };
        }

        let mut left_indices = best_left_indices;
        let mut right_indices = best_right_indices;
        let left_node = self.build_tree(x, y, &mut left_indices, depth + 1, ncols);
        let right_node = self.build_tree(x, y, &mut right_indices, depth + 1, ncols);

        Node {
            feature: best_feature,
            threshold: best_threshold,
            left: Some(Box::new(left_node)),
            right: Some(Box::new(right_node)),
            value: None,
        }
    }

    fn gini_impurity(&self, y: &Array1<f64>, indices: &[usize]) -> f64 {
        let n = indices.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mut counts = std::collections::HashMap::new();
        for &idx in indices {
            let val = y[idx];
            *counts.entry(val.to_bits()).or_insert(0) += 1;
        }
        let mut sum_sq = 0.0;
        for &count in counts.values() {
            let p = (count as f64) / n;
            sum_sq += p * p;
        }
        1.0 - sum_sq
    }

    /// Predict class labels for samples in X.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let root = self.root.as_ref().expect("DecisionTreeClassifier not fitted");
        let (nrows, _) = x.dim();
        let mut preds = Array1::zeros(nrows);
        for r in 0..nrows {
            preds[r] = self.predict_sample(root, x, r);
        }
        preds
    }

    fn predict_sample(&self, node: &Node, x: &Array2<f64>, r: usize) -> f64 {
        if let Some(val) = node.value {
            return val;
        }
        let feat = node.feature.unwrap();
        let thresh = node.threshold.unwrap();
        if x[[r, feat]] <= thresh {
            self.predict_sample(node.left.as_ref().unwrap(), x, r)
        } else {
            self.predict_sample(node.right.as_ref().unwrap(), x, r)
        }
    }
}

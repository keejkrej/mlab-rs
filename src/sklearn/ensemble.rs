use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_classifier() {
        let clf = RandomForestClassifier::new(3);
        let preds = clf.predict(&Array2::zeros((2, 2)));
        assert_eq!(preds.len(), 2);
    }
}

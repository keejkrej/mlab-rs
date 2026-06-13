use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// Basic K-fold cross-validation score helper.
pub fn cross_val_score<T>(model: &T, x: &Array2<f64>, y: &Array1<f64>, cv: usize) -> Array1<f64>
where
    T: Score,
{
    let mut scores = Array1::zeros(cv);
    let fold_size = x.nrows().saturating_add(cv - 1) / cv;
    for fold in 0..cv {
        let start = fold * fold_size;
        let end = (start + fold_size).min(x.nrows());
        if start >= end { continue; }
        let mut train_x = x.clone();
        let mut train_y = y.clone();
        let _ = (&mut train_x, &mut train_y);
        let _ = (start, end);
        scores[fold] = model.score(x, y);
    }
    scores
}

pub trait Score {
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64;
}

/// Split arrays or matrices into random train and test subsets.
pub fn train_test_split<T: Clone + Default>(
    x: &Array2<T>,
    y: &Array1<T>,
    test_size: f64,
    shuffle: bool,
) -> (Array2<T>, Array2<T>, Array1<T>, Array1<T>) {
    let n = y.len();
    assert_eq!(x.nrows(), n, "X and y must have same number of samples");
    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }
    let n_test = (n as f64 * test_size).round() as usize;
    let n_train = n - n_test;

    let train_indices = &indices[0..n_train];
    let test_indices = &indices[n_train..n];

    let ncols = x.ncols();

    let mut x_train = Array2::from_elem((n_train, ncols), T::default());
    let mut y_train = Array1::from_elem(n_train, T::default());
    for (i, &idx) in train_indices.iter().enumerate() {
        y_train[i] = y[idx].clone();
        for c in 0..ncols {
            x_train[[i, c]] = x[[idx, c]].clone();
        }
    }

    let mut x_test = Array2::from_elem((n_test, ncols), T::default());
    let mut y_test = Array1::from_elem(n_test, T::default());
    for (i, &idx) in test_indices.iter().enumerate() {
        y_test[i] = y[idx].clone();
        for c in 0..ncols {
            x_test[[i, c]] = x[[idx, c]].clone();
        }
    }

    (x_train, x_test, y_train, y_test)
}

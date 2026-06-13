pub mod linear_models;
pub mod gmm;
pub mod factorization;

#[cfg(test)]
mod tests {
    use super::factorization::{NMF, VanillaALS};
    use super::gmm::GMM;
    use super::linear_models::{
        BayesianLinearRegressionKnownVariance, BayesianLinearRegressionUnknownVariance,
        GeneralizedLinearModel,
    };
    use ndarray::array;

    #[test]
    fn test_bayesian_linear_regression_known_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = BayesianLinearRegressionKnownVariance::new(0.0, 1.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    fn test_bayesian_linear_regression_unknown_variance() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0];
        let mut model = BayesianLinearRegressionUnknownVariance::new(1.0, 1.0, 0.0, None, true);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[5.0]]);

        assert_eq!(pred.len(), 1);
        assert!(pred[0] > 10.0);
    }

    #[test]
    fn test_generalized_linear_model_identity_link() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mut model = GeneralizedLinearModel::new("identity", true, 1e-6, 25);

        model.fit(&x, &y).unwrap();
        let pred = model.predict(&array![[4.0]]);

        assert_eq!(pred.len(), 1);
        assert!((pred[0] - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_gmm_fit_predict() {
        // Two well-separated clusters in 2D
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.1]
        ];
        let mut model = GMM::new(2, Some(42));
        model.fit(&x, 100, 1e-3, false).unwrap();

        let preds = model.predict(&x, false).unwrap();
        let preds = preds.into_dimensionality::<ndarray::Ix1>().unwrap();

        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_eq!(preds[3], preds[4]);
        assert_eq!(preds[3], preds[5]);
        assert_ne!(preds[0], preds[3]);
    }

    #[test]
    fn test_vanilla_als_factorization() {
        // Rank-2 factorization of a simple product
        let w_true = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let h_true = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = w_true.dot(&h_true);

        let mut als = VanillaALS::new(2, 0.01, 500, 1e-6);
        als.fit(&x, None, None, 10, false).unwrap();

        let (w, h) = als.parameters();
        let w = w.unwrap();
        let h = h.unwrap();
        let x_hat = w.dot(h);
        let mse = (&x - &x_hat).iter().map(|&v| v * v).sum::<f64>() / x.len() as f64;
        assert!(mse < 0.5, "ALS reconstruction MSE too high: {}", mse);
    }

    #[test]
    fn test_nmf_factorization() {
        // Nonnegative rank-2 factorization
        let w_true = array![[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]];
        let h_true = array![[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]];
        let x = w_true.dot(&h_true);

        let mut nmf = NMF::new(2, 500, 1e-6);
        nmf.fit(&x, None, None, 5, false).unwrap();

        let (w, h) = nmf.parameters();
        let w = w.unwrap();
        let h = h.unwrap();

        // Both factors must be nonnegative
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(h.iter().all(|&v| v >= 0.0));

        let x_hat = w.dot(h);
        let mse = (&x - &x_hat).iter().map(|&v| v * v).sum::<f64>() / x.len() as f64;
        assert!(mse < 1.0, "NMF reconstruction MSE too high: {}", mse);
    }
}

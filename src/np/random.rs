use ndarray::Array;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};

/// Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
pub fn rand<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
}

/// Create an array of the given shape and populate it with random samples from a standard normal distribution.
pub fn randn<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    Array::from_shape_simple_fn(shape, || rng.sample(StandardNormal))
}

/// Create an array of the given shape and populate it with random integers from low (inclusive) to high (exclusive).
pub fn randint<Sh>(low: i32, high: i32, shape: Sh) -> Array<i32, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
}

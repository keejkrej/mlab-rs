pub use ndarray::{Array, Array1, Array2, Array3, ArrayD, Axis, Dimension, Ix1, Ix2, Ix3, IxDyn};
pub use ndarray::s;

// Re-expose standard ndarray types for convenience
pub type ArrayN<T, D> = Array<T, D>;

/// Helper trait to construct Array from various types of inputs (vecs, slices, nested vecs)
pub trait IntoArray<T, D: Dimension> {
    fn into_array(self) -> Array<T, D>;
}

impl<T> IntoArray<T, Ix1> for Vec<T> {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self)
    }
}

impl<T: Clone> IntoArray<T, Ix1> for &[T] {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self.to_vec())
    }
}

impl<T: Clone, const N: usize> IntoArray<T, Ix1> for [T; N] {
    fn into_array(self) -> Array<T, Ix1> {
        Array1::from_vec(self.to_vec())
    }
}

// 2D Array from nested Vecs
impl<T> IntoArray<T, Ix2> for Vec<Vec<T>> {
    fn into_array(self) -> Array<T, Ix2> {
        let nrows = self.len();
        if nrows == 0 {
            return Array2::from_shape_vec((0, 0), vec![]).unwrap();
        }
        let ncols = self[0].len();
        let mut flat = Vec::with_capacity(nrows * ncols);
        for mut row in self {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
            flat.append(&mut row);
        }
        Array2::from_shape_vec((nrows, ncols), flat).unwrap()
    }
}

// 2D Array from slice of slices
impl<T: Clone> IntoArray<T, Ix2> for &[&[T]] {
    fn into_array(self) -> Array<T, Ix2> {
        let nrows = self.len();
        if nrows == 0 {
            return Array2::from_shape_vec((0, 0), vec![]).unwrap();
        }
        let ncols = self[0].len();
        let mut flat = Vec::with_capacity(nrows * ncols);
        for row in self {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
            flat.extend_from_slice(row);
        }
        Array2::from_shape_vec((nrows, ncols), flat).unwrap()
    }
}

// Identity conversion for existing arrays
impl<T, D: Dimension> IntoArray<T, D> for Array<T, D> {
    fn into_array(self) -> Array<T, D> {
        self
    }
}

/// Creates a NumPy-like array.
/// Works with vectors, slices, arrays, and nested structures.
pub fn array<T, D: Dimension>(data: impl IntoArray<T, D>) -> Array<T, D> {
    data.into_array()
}

/// Create an array of zeros.
pub fn zeros<T: Clone + num_traits::Zero, Sh>(shape: Sh) -> Array<T, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::zeros(shape)
}

/// Create an array of ones.
pub fn ones<T: Clone + num_traits::One, Sh>(shape: Sh) -> Array<T, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::from_elem(shape, T::one())
}

/// Create an identity matrix.
pub fn eye<T: Clone + num_traits::Zero + num_traits::One>(n: usize) -> Array2<T> {
    Array2::eye(n)
}

/// Create an array filled with zeros with the same shape as the input array.
pub fn zeros_like<T: Clone + num_traits::Zero, D: Dimension>(arr: &Array<T, D>) -> Array<T, D> {
    Array::zeros(arr.raw_dim())
}

/// Create an array filled with ones with the same shape as the input array.
pub fn ones_like<T: Clone + num_traits::One, D: Dimension>(arr: &Array<T, D>) -> Array<T, D> {
    Array::from_elem(arr.raw_dim(), T::one())
}

/// Create a sequence of numbers from start to stop with a given step.
pub fn arange<T>(start: T, stop: T, step: T) -> Array1<T>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let mut values = Vec::new();
    let mut curr = start;
    let step_f = step.to_f64().unwrap();
    if step_f > 0.0 {
        while curr < stop {
            values.push(curr);
            let next_f = curr.to_f64().unwrap() + step_f;
            curr = T::from_f64(next_f).unwrap_or(curr);
        }
    } else if step_f < 0.0 {
        while curr > stop {
            values.push(curr);
            let next_f = curr.to_f64().unwrap() + step_f;
            curr = T::from_f64(next_f).unwrap_or(curr);
        }
    }
    Array1::from_vec(values)
}

/// Create an array of num evenly spaced values from start to stop.
pub fn linspace<T>(start: T, stop: T, num: usize) -> Array1<T>
where
    T: num_traits::Float + num_traits::FromPrimitive,
{
    if num == 0 {
        return Array1::from_vec(vec![]);
    }
    if num == 1 {
        return Array1::from_vec(vec![start]);
    }
    let step = (stop - start) / T::from_usize(num - 1).unwrap();
    Array1::from_shape_fn(num, |i| start + T::from_usize(i).unwrap() * step)
}

/// Random module containing normal and uniform distributions.
pub mod random {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{StandardNormal, Uniform};

    pub fn rand<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
    where
        Sh: ndarray::ShapeBuilder,
    {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0.0, 1.0);
        Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
    }

    pub fn randn<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
    where
        Sh: ndarray::ShapeBuilder,
    {
        let mut rng = rand::thread_rng();
        Array::from_shape_simple_fn(shape, || rng.sample(StandardNormal))
    }

    pub fn randint<Sh>(low: i32, high: i32, shape: Sh) -> Array<i32, Sh::Dim>
    where
        Sh: ndarray::ShapeBuilder,
    {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(low, high);
        Array::from_shape_simple_fn(shape, || dist.sample(&mut rng))
    }
}

// Element-wise Math

pub fn sin<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.sin())
}

pub fn cos<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.cos())
}

pub fn exp<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.exp())
}

pub fn log<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.ln())
}

pub fn sqrt<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Float,
    D: Dimension,
{
    arr.mapv(|x| x.sqrt())
}

pub fn abs<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: num_traits::Signed + Clone,
    D: Dimension,
{
    arr.mapv(|x| x.abs())
}

// Reductions

pub fn sum<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T>,
    D: Dimension,
{
    arr.sum()
}

pub fn sum_axis<T, D>(arr: &Array<T, D>, axis: usize) -> Array<T, <D as Dimension>::Smaller>
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T>,
    D: Dimension + ndarray::RemoveAxis,
{
    arr.sum_axis(Axis(axis))
}

pub fn mean<T, D>(arr: &Array<T, D>) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    arr.mean().unwrap_or_else(|| T::nan())
}

pub fn mean_axis<T, D>(arr: &Array<T, D>, axis: usize) -> Array<T, <D as Dimension>::Smaller>
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension + ndarray::RemoveAxis,
{
    arr.mean_axis(Axis(axis)).unwrap()
}

pub fn var<T, D>(arr: &Array<T, D>, ddof: f64) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    let ddof_t = T::from_f64(ddof).unwrap_or_else(T::zero);
    arr.var(ddof_t)
}

pub fn std<T, D>(arr: &Array<T, D>, ddof: f64) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive,
    D: Dimension,
{
    let ddof_t = T::from_f64(ddof).unwrap_or_else(T::zero);
    arr.std(ddof_t)
}

pub fn min<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + PartialOrd,
    D: Dimension,
{
    arr.iter()
        .fold(arr.first().unwrap().clone(), |a, b| if a < *b { a } else { b.clone() })
}

pub fn max<T, D>(arr: &Array<T, D>) -> T
where
    T: Clone + PartialOrd,
    D: Dimension,
{
    arr.iter()
        .fold(arr.first().unwrap().clone(), |a, b| if a > *b { a } else { b.clone() })
}

pub fn argmin<T, D>(arr: &Array<T, D>) -> usize
where
    T: PartialOrd,
    D: Dimension,
{
    let mut min_idx = 0;
    let mut min_val = arr.first().unwrap();
    for (i, val) in arr.iter().enumerate() {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    min_idx
}

pub fn argmax<T, D>(arr: &Array<T, D>) -> usize
where
    T: PartialOrd,
    D: Dimension,
{
    let mut max_idx = 0;
    let mut max_val = arr.first().unwrap();
    for (i, val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

// Manipulations

pub fn transpose<T, D>(arr: &Array<T, D>) -> Array<T, D>
where
    T: Clone,
    D: Dimension,
{
    arr.t().to_owned()
}

pub fn reshape<T, D, Sh>(arr: &Array<T, D>, shape: Sh) -> Array<T, Sh::Dim>
where
    T: Clone,
    D: Dimension,
    Sh: ndarray::IntoDimension,
{
    arr.to_shape(shape).expect("Shape dimensions must match input array size").to_owned()
}

pub fn concatenate<T, D>(arrays: &[&Array<T, D>], axis: usize) -> Array<T, D>
where
    T: Clone,
    D: Dimension + ndarray::RemoveAxis,
{
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::concatenate(Axis(axis), &views).expect("Failed to concatenate arrays")
}

pub fn stack<T, D>(arrays: &[&Array<T, D>], axis: usize) -> Array<T, D::Larger>
where
    T: Clone,
    D: Dimension + ndarray::RemoveAxis,
    D::Larger: Dimension,
{
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::stack(Axis(axis), &views).expect("Failed to stack arrays")
}

// Linear Algebra

pub fn dot<T>(a: &Array1<T>, b: &Array1<T>) -> T
where
    T: Clone + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(a.len(), b.len(), "Array lengths must match for dot product");
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

pub fn matmul<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: Clone + num_traits::Zero + num_traits::One + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + ndarray::LinalgScalar,
{
    a.dot(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let a: Array1<i32> = array(vec![1, 2, 3]);
        assert_eq!(a.len(), 3);
        assert_eq!(a[0], 1);

        let b: Array2<i32> = array(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(b.dim(), (2, 2));
        assert_eq!(b[[1, 0]], 3);
    }

    #[test]
    fn test_zeros_ones() {
        let z: Array2<f64> = zeros((2, 3));
        assert_eq!(z.dim(), (2, 3));
        assert_eq!(z[[1, 2]], 0.0);

        let o: Array1<i32> = ones(4);
        assert_eq!(o.len(), 4);
        assert_eq!(o[3], 1);
    }

    #[test]
    fn test_arange_linspace() {
        let a = arange(0.0, 5.0, 1.0);
        assert_eq!(a.len(), 5);
        assert_eq!(a[4], 4.0);

        let l = linspace(0.0, 1.0, 5);
        assert_eq!(l.len(), 5);
        assert_eq!(l[4], 1.0);
        assert_eq!(l[2], 0.5);
    }

    #[test]
    fn test_math_and_reductions() {
        let a = array(vec![1.0, 2.0, 3.0]);
        assert_eq!(sum(&a), 6.0);
        assert_eq!(mean(&a), 2.0);
        assert_eq!(min(&a), 1.0);
        assert_eq!(max(&a), 3.0);
        assert_eq!(argmin(&a), 0);
        assert_eq!(argmax(&a), 2);
    }
}


use ndarray::{Array1, Array2};

fn ndarray_to_nalgebra(arr: &Array2<f64>) -> nalgebra::DMatrix<f64> {
    let (nrows, ncols) = arr.dim();
    nalgebra::DMatrix::from_row_iterator(nrows, ncols, arr.iter().cloned())
}

fn nalgebra_to_ndarray(mat: nalgebra::DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = mat.shape();
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in mat.row_iter() {
        flat.extend(row.iter().cloned());
    }
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

pub struct SVDResult {
    pub u: Option<Array2<f64>>,
    pub s: Array1<f64>,
    pub vh: Option<Array2<f64>>,
}

/// Inverse of a matrix.
pub fn inv(arr: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mat = ndarray_to_nalgebra(arr);
    let inv_mat = mat.try_inverse().ok_or_else(|| "Matrix is singular".to_string())?;
    Ok(nalgebra_to_ndarray(inv_mat))
}

/// Determinant of a matrix.
pub fn det(arr: &Array2<f64>) -> f64 {
    let mat = ndarray_to_nalgebra(arr);
    mat.determinant()
}

/// Solve a linear matrix equation, or system of linear scalar equations.
pub fn solve(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let a_mat = ndarray_to_nalgebra(a);
    let b_mat = ndarray_to_nalgebra(b);
    let decomp = a_mat.lu();
    let x_mat = decomp.solve(&b_mat).ok_or_else(|| "Matrix solver failed".to_string())?;
    Ok(nalgebra_to_ndarray(x_mat))
}

/// Solve a linear system with a 1D vector target.
pub fn solve_vec(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let a_mat = ndarray_to_nalgebra(a);
    let b_vec = nalgebra::DVector::from_iterator(b.len(), b.iter().cloned());
    let decomp = a_mat.lu();
    let x_vec = decomp.solve(&b_vec).ok_or_else(|| "Matrix solver failed".to_string())?;
    Ok(Array1::from_vec(x_vec.iter().cloned().collect()))
}

/// Cholesky decomposition.
pub fn cholesky(arr: &Array2<f64>, lower: bool) -> Result<Array2<f64>, String> {
    let mat = ndarray_to_nalgebra(arr);
    let chol = mat.cholesky().ok_or_else(|| "Matrix is not positive-definite".to_string())?;
    let l_mat = chol.unpack();
    if lower {
        Ok(nalgebra_to_ndarray(l_mat))
    } else {
        Ok(nalgebra_to_ndarray(l_mat.transpose()))
    }
}

/// Singular Value Decomposition.
pub fn svd(arr: &Array2<f64>) -> SVDResult {
    let mat = ndarray_to_nalgebra(arr);
    let svd = mat.svd(true, true);
    let u = svd.u.map(|u_mat| nalgebra_to_ndarray(u_mat));
    let s = Array1::from_vec(svd.singular_values.iter().cloned().collect());
    let vh = svd.v_t.map(|vt_mat| nalgebra_to_ndarray(vt_mat));
    SVDResult { u, s, vh }
}

/// Compute singular values of a matrix.
pub fn svdvals(arr: &Array2<f64>) -> Array1<f64> {
    let mat = ndarray_to_nalgebra(arr);
    let svd = mat.svd(false, false);
    Array1::from_vec(svd.singular_values.iter().cloned().collect())
}

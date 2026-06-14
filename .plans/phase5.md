# Phase 5: Advanced & Niche

Advanced scientific computing and specialized algorithms.

## sp.integrate -- numerical integration

- [ ] `quad` -- 1D quadrature (adaptive Gauss-Kronrod)
- [ ] `dblquad` -- 2D quadrature
- [ ] `trapz` / `simpson` -- trapezoidal/Simpson's rule
- [ ] `cumtrapz` -- cumulative trapezoidal
- [ ] `solve_ivp` -- ODE initial value solver (RK45, RK23, Radau, BDF)
- [ ] `odeint` -- ODE integrator (scipy-compatible)
- [ ] `romberg` -- Romberg integration

## sp.sparse -- sparse matrix operations

- [ ] `csr_matrix` -- compressed sparse row
- [ ] `csc_matrix` -- compressed sparse column
- [ ] `coo_matrix` -- coordinate format
- [ ] Sparse arithmetic (+, -, *, dot)
- [ ] `linalg.spsolve` -- sparse linear solve
- [ ] `linalg.eigsh` -- sparse eigenvalue solver
- [ ] `linalg.norm` -- sparse norm

## sp.spatial -- spatial data structures

- [ ] `KDTree` -- k-d tree for nearest neighbor
- [ ] `pdist` -- pairwise distances (condensed form)
- [ ] `squareform` -- convert between condensed and square
- [ ] `ConvexHull` -- convex hull
- [ ] `Voronoi` -- Voronoi tessellation
- [ ] `Delaunay` -- Delaunay triangulation
- [ ] Distance functions: `euclidean`, `manhattan`, `cosine`, `chebyshev`, `minkowski`

## sklearn.manifold -- dimensionality reduction

- [ ] `TSNE` -- t-distributed stochastic neighbor embedding
- [ ] `Isomap` -- isometric mapping

## sklearn.feature_selection -- feature selection

- [ ] `SelectKBest` -- select top k features
- [ ] `RFE` -- recursive feature elimination
- [ ] `mutual_info_classif` / `mutual_info_reg` -- mutual information

## sklearn.gaussian_process -- Gaussian processes

- [ ] `GaussianProcessRegressor` -- GP regression
- [ ] `GaussianProcessClassifier` -- GP classification

## sklearn.mixture -- mixture models

- [ ] `GaussianMixture` -- Gaussian mixture model (sklearn interface)

## sklearn.discriminant_analysis

- [ ] `LinearDiscriminantAnalysis` -- LDA
- [ ] `QuadraticDiscriminantAnalysis` -- QDA

## sklearn.decomposition -- additional methods

- [ ] `NMF` -- non-negative matrix factorization
- [ ] `TruncatedSVD` -- truncated SVD (for sparse)
- [ ] `FactorAnalysis` -- factor analysis

## sp.special -- additional special functions

- [ ] `iv`, `jv` -- Bessel functions (arbitrary order)
- [ ] `kv`, `yv` -- modified Bessel of 2nd kind, Bessel of 2nd kind
- [ ] `ndtr` / `ndtri` -- normal CDF and its inverse
- [ ] `hyp1f1`, `hyp2f1` -- hypergeometric functions
- [ ] `ellipk`, `ellipe` -- complete elliptic integrals
- [ ] `airy` -- Airy functions
- [ ] `sph_harm` -- spherical harmonics

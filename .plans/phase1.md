# Phase 1: Foundational Gaps

Blocks everything else. These are the building blocks used by all higher-level code.

## sp.linalg -- add missing decompositions and solvers

- [ ] `eig` -- eigenvalue decomposition (real non-symmetric)
- [ ] `eigh` -- eigenvalue decomposition (symmetric/Hermitian)
- [ ] `lstsq` -- least-squares solution to Ax = b
- [ ] `qr` -- QR decomposition
- [ ] `lu` / `lu_factor` / `lu_solve` -- LU decomposition + solve
- [ ] `pinv` -- pseudo-inverse (Moore-Penrose)
- [ ] `norm` -- matrix and vector norms (1, 2, inf, Frobenius)
- [ ] `matrix_rank` -- rank via SVD
- [ ] `expm` -- matrix exponential (Padé approximant)
- [ ] `kron` -- Kronecker product
- [ ] `solve_triangular` -- solve with triangular matrix
- [ ] `block_diag` -- block diagonal matrix construction
- [ ] `null_space` -- null space of a matrix
- [ ] `orth` -- orthogonal basis for range of matrix

## sp.special -- special mathematical functions

- [ ] `gamma` -- Gamma function Γ(x)
- [ ] `gammaln` -- log of Gamma function
- [ ] `digamma` / `psi` -- digamma function ψ(x)
- [ ] `beta` -- Beta function B(a,b)
- [ ] `betaln` -- log of Beta function
- [ ] `erfcx` -- scaled complementary error function
- [ ] `erfinv` -- inverse error function
- [ ] `expit` -- logistic sigmoid 1/(1+exp(-x))
- [ ] `logit` -- logit function log(p/(1-p))
- [ ] `logsumexp` -- stable log of sum of exponentials
- [ ] `softmax` -- numerically stable softmax
- [ ] `softmax` -- numerically stable softmax
- [ ] `bessel_i0`, `bessel_i1` -- modified Bessel functions
- [ ] `bessel_j0`, `bessel_j1` -- Bessel functions of first kind
- [ ] `factorial` / `factorialln` -- factorial and log factorial
- [ ] `comb` / `perm` -- combinations and permutations
- [ ] `poch` -- Pochhammer symbol (rising factorial)

## sp.stats -- distributions and hypothesis tests

### Distribution objects (trait + implementations)
- [ ] `Distribution` trait: `pdf`, `cdf`, `ppf` (quantile), `sf` (survival), `isf`, `rvs` (random), `mean`, `var`, `std`, `median`, `interval`
- [ ] `Norm` distribution (extend existing)
- [ ] `T` distribution (Student's t)
- [ ] `Chi2` distribution (chi-squared)
- [ ] `F` distribution
- [ ] `Beta` distribution
- [ ] `Gamma` distribution
- [ ] `Exponential` distribution
- [ ] `Poisson` distribution
- [ ] `Binomial` distribution
- [ ] `Uniform` distribution
- [ ] `LogNormal` distribution
- [ ] `Bernoulli` distribution

### Hypothesis tests
- [ ] `ttest_1samp` -- one-sample t-test
- [ ] `ttest_ind` -- independent two-sample t-test
- [ ] `ttest_rel` -- paired t-test
- [ ] `f_oneway` -- one-way ANOVA
- [ ] `mannwhitneyu` -- Mann-Whitney U test
- [ ] `wilcoxon` -- Wilcoxon signed-rank test
- [ ] `ks_2samp` -- Kolmogorov-Smirnov two-sample test
- [ ] `shapiro` -- Shapiro-Wilk normality test
- [ ] `chi2_contingency` -- chi-squared test of independence

### Descriptive statistics
- [ ] `kurtosis` -- excess kurtosis
- [ ] `skew` -- skewness
- [ ] `describe` -- summary statistics
- [ ] `mode` -- mode of array
- [ ] `iqr` -- interquartile range
- [ ] `sem` -- standard error of the mean
- [ ] `linregress` -- linear regression (slope, intercept, r, p, stderr)
- [ ] `entropy` -- Shannon entropy
- [ ] `wasserstein_distance` -- 1D Wasserstein distance

## np.random -- additional distributions and utilities

- [ ] `uniform` -- uniform distribution draws
- [ ] `normal` -- normal distribution draws
- [ ] `choice` -- random sampling from array
- [ ] `shuffle` -- in-place shuffle
- [ ] `permutation` -- return shuffled copy
- [ ] `seed` -- set RNG seed
- [ ] `exponential` -- exponential distribution
- [ ] `poisson` -- Poisson distribution
- [ ] `binomial` -- binomial distribution
- [ ] `beta` -- beta distribution
- [ ] `gamma` -- gamma distribution
- [ ] `multivariate_normal` -- multivariate normal

## sp.fft -- missing FFT utilities

- [ ] `rfft` -- real-input FFT (returns positive frequencies only)
- [ ] `irfft` -- inverse of rfft
- [ ] `fftfreq` -- sample frequencies for FFT
- [ ] `rfftfreq` -- sample frequencies for rfft
- [ ] `fftshift` -- shift zero-frequency component to center
- [ ] `ifftshift` -- inverse of fftshift

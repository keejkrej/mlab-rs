use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// DBSCAN density-based clustering.
pub struct DBSCAN {
    pub eps: f64,
    pub min_samples: usize,
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    pub fn fit_predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let nrows = x.nrows();
        let ncols = x.ncols();
        let eps_sq = self.eps * self.eps;

        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut nbrs = Vec::new();
            for j in 0..nrows {
                if i == j {
                    continue;
                }
                let mut dist_sq = 0.0;
                for c in 0..ncols {
                    dist_sq += (x[[i, c]] - x[[j, c]]).powi(2);
                }
                if dist_sq <= eps_sq {
                    nbrs.push(j);
                }
            }
            neighbors.push(nbrs);
        }

        let mut labels = Array1::from_elem(nrows, -1.0f64);
        let mut visited = vec![false; nrows];
        let mut cluster_id = 0.0f64;

        for i in 0..nrows {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            if neighbors[i].len() < self.min_samples {
                continue;
            }

            labels[i] = cluster_id;
            let mut seed_set: Vec<usize> = neighbors[i].clone();
            let mut idx = 0;
            while idx < seed_set.len() {
                let q = seed_set[idx];
                if !visited[q] {
                    visited[q] = true;
                    if neighbors[q].len() >= self.min_samples {
                        for &nbr in &neighbors[q] {
                            if !seed_set.contains(&nbr) {
                                seed_set.push(nbr);
                            }
                        }
                    }
                }
                if labels[q] < 0.0 {
                    labels[q] = cluster_id;
                }
                idx += 1;
            }

            cluster_id += 1.0;
        }

        labels
    }
}

/// K-Means clustering.
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub cluster_centers: Option<Array2<f64>>,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol: 1e-4,
            cluster_centers: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (nrows, ncols) = x.dim();
        if nrows < self.n_clusters {
            panic!("Number of samples must be >= n_clusters");
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..nrows).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Array2::zeros((self.n_clusters, ncols));
        for k in 0..self.n_clusters {
            let idx = indices[k];
            for c in 0..ncols {
                centroids[[k, c]] = x[[idx, c]];
            }
        }

        let mut labels = vec![0; nrows];

        for _ in 0..self.max_iter {
            let mut changed = false;
            for r in 0..nrows {
                let mut min_dist = f64::INFINITY;
                let mut best_k = 0;
                for k in 0..self.n_clusters {
                    let mut dist_sq = 0.0;
                    for c in 0..ncols {
                        dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                    }
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_k = k;
                    }
                }
                if labels[r] != best_k {
                    labels[r] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            let mut new_centroids: Array2<f64> = Array2::zeros((self.n_clusters, ncols));
            let mut counts = vec![0.0; self.n_clusters];
            for r in 0..nrows {
                let k = labels[r];
                counts[k] += 1.0;
                for c in 0..ncols {
                    new_centroids[[k, c]] += x[[r, c]];
                }
            }

            let mut centroid_diff: f64 = 0.0;
            for k in 0..self.n_clusters {
                if counts[k] > 0.0 {
                    for c in 0..ncols {
                        let new_val = new_centroids[[k, c]] / counts[k];
                        let diff = centroids[[k, c]] - new_val;
                        centroid_diff += diff * diff;
                        centroids[[k, c]] = new_val;
                    }
                }
            }

            if centroid_diff.sqrt() < self.tol {
                break;
            }
        }

        self.cluster_centers = Some(centroids);
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let centroids = self.cluster_centers.as_ref().expect("KMeans model not fitted");
        let (nrows, ncols) = x.dim();
        let mut labels = Array1::zeros(nrows);
        for r in 0..nrows {
            let mut min_dist = f64::INFINITY;
            let mut best_k = 0;
            for k in 0..self.n_clusters {
                let mut dist_sq = 0.0;
                for c in 0..ncols {
                    dist_sq += (x[[r, c]] - centroids[[k, c]]).powi(2);
                }
                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    best_k = k;
                }
            }
            labels[r] = best_k;
        }
        labels
    }
}

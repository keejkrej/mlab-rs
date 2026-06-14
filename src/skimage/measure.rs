use ndarray::Array2;

/// Properties of a labeled region.
pub struct RegionProps {
    pub label: usize,
    pub area: usize,
    pub bbox: (usize, usize, usize, usize), // min_row, min_col, max_row, max_col
    pub centroid: (f64, f64),
    pub perimeter: f64,
}

/// Compute properties for each labeled region.
pub fn regionprops(labels: &Array2<usize>, num_labels: usize) -> Vec<RegionProps> {
    let (height, width) = labels.dim();
    let mut areas = vec![0usize; num_labels + 1];
    let mut sum_r = vec![0usize; num_labels + 1];
    let mut sum_c = vec![0usize; num_labels + 1];
    let mut min_r = vec![usize::MAX; num_labels + 1];
    let mut min_c = vec![usize::MAX; num_labels + 1];
    let mut max_r = vec![0usize; num_labels + 1];
    let mut max_c = vec![0usize; num_labels + 1];

    for r in 0..height {
        for c in 0..width {
            let lab = labels[[r, c]];
            if lab == 0 {
                continue;
            }
            areas[lab] += 1;
            sum_r[lab] += r;
            sum_c[lab] += c;
            if r < min_r[lab] { min_r[lab] = r; }
            if c < min_c[lab] { min_c[lab] = c; }
            if r > max_r[lab] { max_r[lab] = r; }
            if c > max_c[lab] { max_c[lab] = c; }
        }
    }

    let mut props = Vec::new();
    for lab in 1..=num_labels {
        if areas[lab] == 0 {
            continue;
        }
        let mask = labels.mapv(|v| v == lab);
        let perim = perimeter(&mask);
        props.push(RegionProps {
            label: lab,
            area: areas[lab],
            bbox: (min_r[lab], min_c[lab], max_r[lab] + 1, max_c[lab] + 1),
            centroid: (
                sum_r[lab] as f64 / areas[lab] as f64,
                sum_c[lab] as f64 / areas[lab] as f64,
            ),
            perimeter: perim,
        });
    }
    props
}

/// Find iso-level contours using the marching squares algorithm.
/// Returns a list of contours, each a list of (row, col) points.
pub fn find_contours(image: &Array2<f64>, level: f64) -> Vec<Vec<(f64, f64)>> {
    let (height, width) = image.dim();
    if height < 2 || width < 2 {
        return vec![];
    }

    let mut segments: Vec<((f64, f64), (f64, f64))> = Vec::new();

    for r in 0..height - 1 {
        for c in 0..width - 1 {
            let tl = image[[r, c]];
            let tr = image[[r, c + 1]];
            let br = image[[r + 1, c + 1]];
            let bl = image[[r + 1, c]];

            let mut case = 0u8;
            if tl >= level { case |= 1; }
            if tr >= level { case |= 2; }
            if br >= level { case |= 4; }
            if bl >= level { case |= 8; }

            if case == 0 || case == 15 {
                continue;
            }

            let interp = |v0: f64, v1: f64| -> f64 {
                let d = v1 - v0;
                if d.abs() < 1e-12 {
                    0.5
                } else {
                    (level - v0) / d
                }
            };

            let r_f = r as f64;
            let c_f = c as f64;

            let top = (r_f, c_f + interp(tl, tr));
            let right = (r_f + interp(tr, br), c_f + 1.0);
            let bottom = (r_f + 1.0, c_f + interp(bl, br));
            let left = (r_f + interp(tl, bl), c_f);

            let add_segment = |segments: &mut Vec<((f64, f64), (f64, f64))>, a: (f64, f64), b: (f64, f64)| {
                segments.push((a, b));
            };

            match case {
                1 | 14 => add_segment(&mut segments, top, left),
                2 | 13 => add_segment(&mut segments, top, right),
                3 | 12 => add_segment(&mut segments, left, right),
                4 | 11 => add_segment(&mut segments, right, bottom),
                5 => {
                    add_segment(&mut segments, top, right);
                    add_segment(&mut segments, left, bottom);
                }
                6 | 9 => add_segment(&mut segments, top, bottom),
                7 | 8 => add_segment(&mut segments, left, bottom),
                10 => {
                    add_segment(&mut segments, top, left);
                    add_segment(&mut segments, right, bottom);
                }
                _ => {}
            }
        }
    }

    if segments.is_empty() {
        return vec![];
    }

    let mut used = vec![false; segments.len()];
    let mut contours: Vec<Vec<(f64, f64)>> = Vec::new();

    for i in 0..segments.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut chain = vec![segments[i].0, segments[i].1];
        let eps = 1e-9;

        loop {
            let tail = *chain.last().unwrap();
            let mut found = false;
            for j in 0..segments.len() {
                if used[j] {
                    continue;
                }
                let (a, b) = segments[j];
                if (a.0 - tail.0).abs() < eps && (a.1 - tail.1).abs() < eps {
                    used[j] = true;
                    chain.push(b);
                    found = true;
                    break;
                }
                if (b.0 - tail.0).abs() < eps && (b.1 - tail.1).abs() < eps {
                    used[j] = true;
                    chain.push(a);
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        contours.push(chain);
    }

    contours
}

/// 4-connected component labeling on a binary image.
/// Returns (labeled_array, num_labels).
pub fn label(input: &Array2<bool>) -> (Array2<usize>, usize) {
    let (height, width) = input.dim();
    let mut labels = Array2::zeros((height, width));
    let mut current_label = 0usize;
    let mut equivalences: Vec<usize> = vec![0];

    fn find_root(equiv: &mut [usize], mut x: usize) -> usize {
        while equiv[x] != x {
            equiv[x] = equiv[equiv[x]];
            x = equiv[x];
        }
        x
    }

    fn union(equiv: &mut [usize], a: usize, b: usize) {
        let ra = find_root(equiv, a);
        let rb = find_root(equiv, b);
        if ra != rb {
            if ra < rb {
                equiv[rb] = ra;
            } else {
                equiv[ra] = rb;
            }
        }
    }

    for r in 0..height {
        for c in 0..width {
            if !input[[r, c]] {
                continue;
            }

            let above = if r > 0 && input[[r - 1, c]] {
                Some(labels[[r - 1, c]])
            } else {
                None
            };
            let left = if c > 0 && input[[r, c - 1]] {
                Some(labels[[r, c - 1]])
            } else {
                None
            };

            match (above, left) {
                (None, None) => {
                    current_label += 1;
                    equivalences.push(current_label);
                    labels[[r, c]] = current_label;
                }
                (Some(a), None) => {
                    labels[[r, c]] = a;
                }
                (None, Some(l)) => {
                    labels[[r, c]] = l;
                }
                (Some(a), Some(l)) => {
                    labels[[r, c]] = a.min(l);
                    union(&mut equivalences, a, l);
                }
            }
        }
    }

    for lab in 1..=current_label {
        let _ = find_root(&mut equivalences, lab);
    }

    let mut roots: Vec<usize> = vec![0; current_label + 1];
    for i in 1..=current_label {
        roots[i] = find_root(&mut equivalences, i);
    }

    let mut unique_roots: Vec<usize> = roots[1..].to_vec();
    unique_roots.sort_unstable();
    unique_roots.dedup();

    let mut remap = vec![0usize; current_label + 1];
    for (new_label, &root) in unique_roots.iter().enumerate() {
        remap[root] = new_label + 1;
    }
    for i in 1..=current_label {
        remap[i] = remap[roots[i]];
    }

    labels.mapv_inplace(|v| {
        if v == 0 { 0 } else { remap[v] }
    });

    let num_labels = unique_roots.len();
    (labels, num_labels)
}

/// Compute raw moments M_pq = sum(x^p * y^q * I(x,y)).
/// Returns a 4x4 matrix for p,q in 0..4.
pub fn moments(image: &Array2<f64>) -> [[f64; 4]; 4] {
    let (height, width) = image.dim();
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..height {
        for c in 0..width {
            let v = image[[r, c]];
            let y = r as f64;
            let x = c as f64;
            for p in 0..4 {
                for q in 0..4 {
                    m[p][q] += x.powi(p as i32) * y.powi(q as i32) * v;
                }
            }
        }
    }
    m
}

/// Compute central moments mu_pq.
pub fn moments_central(image: &Array2<f64>, center: (f64, f64)) -> [[f64; 4]; 4] {
    let (height, width) = image.dim();
    let (cx, cy) = center;
    let mut mu = [[0.0f64; 4]; 4];
    for r in 0..height {
        for c in 0..width {
            let v = image[[r, c]];
            let dy = r as f64 - cy;
            let dx = c as f64 - cx;
            for p in 0..4 {
                for q in 0..4 {
                    mu[p][q] += dx.powi(p as i32) * dy.powi(q as i32) * v;
                }
            }
        }
    }
    mu
}

/// Compute Hu's 7 rotation-invariant moments from a moments matrix.
pub fn moments_hu(m: &[[f64; 4]; 4]) -> [f64; 7] {
    let m00 = m[0][0];
    if m00.abs() < 1e-12 {
        return [0.0; 7];
    }

    let cx = m[1][0] / m00;
    let cy = m[0][1] / m00;

    let nu = |p: usize, q: usize| -> f64 {
        let raw = m[p][q];
        let mut val = raw;
        for k in 0..=p {
            for j in 0..=q {
                let coeff = {
                    let bp = binom(p, k);
                    let bq = binom(q, j);
                    bp * bq
                };
                if (k + j) % 2 == 1 {
                    val -= coeff as f64 * cx.powi(k as i32) * cy.powi(j as i32) * m[p - k][q - j];
                } else if k + j > 0 {
                    val += coeff as f64 * cx.powi(k as i32) * cy.powi(j as i32) * m[p - k][q - j];
                }
            }
        }
        val / m00.powf(1.0 + (p + q) as f64 / 2.0)
    };

    let n20 = nu(2, 0);
    let n02 = nu(0, 2);
    let n11 = nu(1, 1);
    let n30 = nu(3, 0);
    let n12 = nu(1, 2);
    let n21 = nu(2, 1);
    let n03 = nu(0, 3);

    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11.powi(2);
    let h3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let h4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    let h5 = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));
    let h6 = (n20 - n02) * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
        + 4.0 * n11 * (n30 + n12) * (n21 + n03);
    let h7 = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));

    [h1, h2, h3, h4, h5, h6, h7]
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute weighted centroid (row, col) of an image.
pub fn centroid(image: &Array2<f64>) -> (f64, f64) {
    let m = moments(image);
    let m00 = m[0][0];
    if m00.abs() < 1e-12 {
        let (h, w) = image.dim();
        return ((h - 1) as f64 / 2.0, (w - 1) as f64 / 2.0);
    }
    (m[0][1] / m00, m[1][0] / m00)
}

/// Count boundary pixels (pixels with at least one non-neighbor).
pub fn perimeter(labels: &Array2<bool>) -> f64 {
    let (height, width) = labels.dim();
    let mut count = 0usize;
    for r in 0..height {
        for c in 0..width {
            if !labels[[r, c]] {
                continue;
            }
            let mut on_boundary = false;
            for &(dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                    on_boundary = true;
                    break;
                }
                if !labels[[nr as usize, nc as usize]] {
                    on_boundary = true;
                    break;
                }
            }
            if on_boundary {
                count += 1;
            }
        }
    }
    count as f64
}

/// Compute 2x2 inertia tensor from central moments.
pub fn inertia_tensor(image: &Array2<f64>) -> [[f64; 2]; 2] {
    let (cy, cx) = centroid(image);
    let mu = moments_central(image, (cx, cy));
    let mu20 = mu[2][0];
    let mu02 = mu[0][2];
    let mu11 = mu[1][1];
    [[mu02, -mu11], [-mu11, mu20]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regionprops_single_square() {
        let mut labels = Array2::zeros((10, 10));
        for r in 2..5 {
            for c in 3..7 {
                labels[[r, c]] = 1;
            }
        }
        let props = regionprops(&labels, 1);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].area, 12);
        assert_eq!(props[0].bbox, (2, 3, 5, 7));
        assert!((props[0].centroid.0 - 3.0).abs() < 1e-9);
        assert!((props[0].centroid.1 - 4.5).abs() < 1e-9);
    }

    #[test]
    fn test_find_contours_circle() {
        let size = 20usize;
        let mut image = Array2::zeros((size, size));
        let center = 9.5f64;
        let radius = 5.0f64;
        for r in 0..size {
            for c in 0..size {
                let d = ((r as f64 - center).powi(2) + (c as f64 - center).powi(2)).sqrt();
                if d <= radius {
                    image[[r, c]] = 1.0;
                }
            }
        }
        let contours = find_contours(&image, 0.5);
        assert!(!contours.is_empty());
        for contour in &contours {
            assert!(contour.len() >= 3);
            for &(r, c) in contour {
                let d = ((r - center).powi(2) + (c - center).powi(2)).sqrt();
                assert!((d - radius).abs() < 2.0, "contour point too far from circle edge");
            }
        }
    }

    #[test]
    fn test_moments_symmetric() {
        let mut image = Array2::zeros((11, 11));
        for r in 3..8 {
            for c in 3..8 {
                image[[r, c]] = 1.0;
            }
        }
        let m = moments(&image);
        let hu = moments_hu(&m);
        assert!(hu[1].abs() < 1e-6, "h2 should be ~0 for symmetric shape");
        assert!(hu[3].abs() < 1e-6, "h4 should be ~0 for symmetric shape");
        assert!(hu[5].abs() < 1e-6, "h6 should be ~0 for symmetric shape");
    }

    #[test]
    fn test_label_two_components() {
        let mut input = Array2::from_elem((5, 5), false);
        for r in 0..2 {
            for c in 0..2 {
                input[[r, c]] = true;
            }
        }
        for r in 3..5 {
            for c in 3..5 {
                input[[r, c]] = true;
            }
        }
        let (labeled, num) = label(&input);
        assert_eq!(num, 2);
        assert_eq!(labeled[[0, 0]], labeled[[1, 1]]);
        assert_eq!(labeled[[3, 3]], labeled[[4, 4]]);
        assert_ne!(labeled[[0, 0]], labeled[[3, 3]]);
    }

    #[test]
    fn test_centroid_symmetric() {
        let mut image = Array2::zeros((5, 5));
        for r in 1..4 {
            for c in 1..4 {
                image[[r, c]] = 1.0;
            }
        }
        let (cr, cc) = centroid(&image);
        assert!((cr - 2.0).abs() < 1e-9);
        assert!((cc - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_perimeter_square() {
        let mut mask = Array2::from_elem((5, 5), false);
        for r in 1..4 {
            for c in 1..4 {
                mask[[r, c]] = true;
            }
        }
        let p = perimeter(&mask);
        assert_eq!(p, 8.0);
    }

    #[test]
    fn test_inertia_tensor_symmetric() {
        let mut image = Array2::zeros((7, 7));
        for r in 2..5 {
            for c in 2..5 {
                image[[r, c]] = 1.0;
            }
        }
        let it = inertia_tensor(&image);
        assert!((it[0][1]).abs() < 1e-9);
        assert!((it[1][0]).abs() < 1e-9);
        assert!((it[0][0] - it[1][1]).abs() < 1e-9);
    }
}

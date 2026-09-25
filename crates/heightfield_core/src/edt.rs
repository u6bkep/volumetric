//! Exact Euclidean distance transforms on the grid (Felzenszwalb &
//! Huttenlocher's separable lower envelope of parabolas), in cell units.

/// 1D squared-distance transform of `f` (INFINITY = no site) into `out`,
/// with the arg-min site index in `arg`.
fn dt1(f: &[f64], out: &mut [f64], arg: &mut [usize], v: &mut Vec<usize>, z: &mut Vec<f64>) {
    let n = f.len();
    v.clear();
    z.clear();
    v.resize(n, 0);
    z.resize(n + 1, 0.0);
    let mut k = 0usize;
    let mut started = false;
    for q in 0..n {
        if f[q] == f64::INFINITY {
            continue;
        }
        if !started {
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            started = true;
            continue;
        }
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
            if s <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
                continue;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    if !started {
        for q in 0..n {
            out[q] = f64::INFINITY;
            arg[q] = usize::MAX;
        }
        return;
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let d = q as f64 - p as f64;
        out[q] = d * d + f[p];
        arg[q] = p;
    }
}

/// Squared Euclidean distance (in cells) from every cell to the nearest
/// `site` cell, and that site's index. Cells with no site anywhere get
/// `INFINITY` / `usize::MAX`.
pub fn distance_to_sites(nx: usize, ny: usize, site: impl Fn(usize) -> bool) -> (Vec<f64>, Vec<usize>) {
    let n = nx * ny;
    let mut d = vec![f64::INFINITY; n];
    let mut nearest = vec![usize::MAX; n];
    for i in 0..n {
        if site(i) {
            d[i] = 0.0;
            nearest[i] = i;
        }
    }
    let mut v = Vec::new();
    let mut z = Vec::new();
    // Columns.
    let mut f = vec![0.0; ny];
    let mut o = vec![0.0; ny];
    let mut a = vec![0usize; ny];
    for i in 0..nx {
        for j in 0..ny {
            f[j] = d[j * nx + i];
        }
        dt1(&f, &mut o, &mut a, &mut v, &mut z);
        for j in 0..ny {
            d[j * nx + i] = o[j];
            nearest[j * nx + i] = if a[j] == usize::MAX { usize::MAX } else { a[j] * nx + i };
        }
    }
    // Rows.
    let mut f = vec![0.0; nx];
    let mut o = vec![0.0; nx];
    let mut a = vec![0usize; nx];
    let mut near_row = vec![usize::MAX; nx];
    for j in 0..ny {
        for i in 0..nx {
            f[i] = d[j * nx + i];
            near_row[i] = nearest[j * nx + i];
        }
        dt1(&f, &mut o, &mut a, &mut v, &mut z);
        for i in 0..nx {
            d[j * nx + i] = o[i];
            nearest[j * nx + i] = if a[i] == usize::MAX { usize::MAX } else { near_row[a[i]] };
        }
    }
    (d, nearest)
}

/// Signed distance in cells to the boundary of `inside` (positive inside):
/// the distance to the nearest outside cell minus half a cell inside, and
/// the negative of the distance to the nearest inside cell plus half a cell
/// outside, so that the zero crossing lies between the two cell centres.
pub fn signed_distance(nx: usize, ny: usize, inside: &[bool]) -> Vec<f64> {
    let (d_out, _) = distance_to_sites(nx, ny, |i| !inside[i]);
    let (d_in, _) = distance_to_sites(nx, ny, |i| inside[i]);
    (0..nx * ny)
        .map(|i| if inside[i] { d_out[i].sqrt() - 0.5 } else { -(d_in[i].sqrt() - 0.5) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_match_brute_force() {
        let (nx, ny) = (17, 11);
        let sites: Vec<bool> = (0..nx * ny).map(|i| (i * 7919) % 13 == 0).collect();
        let (d, near) = distance_to_sites(nx, ny, |i| sites[i]);
        for j in 0..ny {
            for i in 0..nx {
                let mut best = f64::INFINITY;
                for jj in 0..ny {
                    for ii in 0..nx {
                        if sites[jj * nx + ii] {
                            let dd = ((i as f64 - ii as f64).powi(2)) + ((j as f64 - jj as f64).powi(2));
                            best = best.min(dd);
                        }
                    }
                }
                assert_eq!(d[j * nx + i], best);
                let k = near[j * nx + i];
                let (ki, kj) = (k % nx, k / nx);
                let dd = (i as f64 - ki as f64).powi(2) + (j as f64 - kj as f64).powi(2);
                assert_eq!(dd, best);
                assert!(sites[k]);
            }
        }
    }

    #[test]
    fn signed_distance_of_a_square() {
        let (nx, ny) = (20, 20);
        let inside: Vec<bool> = (0..nx * ny).map(|i| (5..15).contains(&(i % nx)) && (5..15).contains(&(i / nx))).collect();
        let s = signed_distance(nx, ny, &inside);
        assert_eq!(s[10 * nx + 10], 4.5);
        assert_eq!(s[10 * nx + 5], 0.5);
        assert_eq!(s[10 * nx + 4], -0.5);
        assert_eq!(s[0], -((5.0f64 * 5.0 * 2.0).sqrt() - 0.5));
    }
}

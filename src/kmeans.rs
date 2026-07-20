//! k-means clustering primitives — the compute-heavy hot loops for distributed
//! Lloyd's algorithm, GIL-free in Rust.
//!
//! Distributed k-means is map-reduce: each worker assigns its shard's points to
//! the nearest centroid and accumulates per-cluster (sum, count); the driver
//! merges the partial sums into new centroids; iterate. The per-point
//! nearest-centroid scan is the hot loop — here in Rust. The Python driver owns
//! the iteration + the (small) centroid merge.
//!
//! Used to scale semantic dedup (cluster embeddings, then dedup within each
//! cluster — avoids a global O(n^2) comparison) and as the basis of IVF-style
//! bucketing.

/// Squared Euclidean distance between two equal-length vectors.
fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Index of the nearest centroid to `point` (by squared Euclidean distance).
pub fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sqdist(point, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// One local map step of Lloyd's algorithm over a shard: assign each point to
/// its nearest centroid and accumulate per-cluster coordinate sums + counts.
/// Returns (sums[k][dim], counts[k], inertia) — the partial statistics the
/// driver merges across shards. `points` is row-major n×dim flat.
pub fn assign_accumulate(
    points: &[f32],
    n: usize,
    dim: usize,
    centroids: &[Vec<f32>],
) -> (Vec<Vec<f64>>, Vec<u64>, f64) {
    let k = centroids.len();
    let mut sums = vec![vec![0f64; dim]; k];
    let mut counts = vec![0u64; k];
    let mut inertia = 0f64;
    for i in 0..n {
        let p = &points[i * dim..(i + 1) * dim];
        let c = nearest_centroid(p, centroids);
        counts[c] += 1;
        let cs = &mut sums[c];
        for j in 0..dim {
            cs[j] += p[j] as f64;
        }
        inertia += sqdist(p, &centroids[c]) as f64;
    }
    (sums, counts, inertia)
}

/// Just the assignment (nearest centroid per point) — for the final labelling
/// pass or for bucketing after centroids are fixed.
pub fn assign_labels(points: &[f32], n: usize, dim: usize, centroids: &[Vec<f32>]) -> Vec<u32> {
    (0..n)
        .map(|i| nearest_centroid(&points[i * dim..(i + 1) * dim], centroids) as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_is_correct() {
        let cents = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        assert_eq!(nearest_centroid(&[1.0, 1.0], &cents), 0);
        assert_eq!(nearest_centroid(&[9.0, 8.0], &cents), 1);
    }

    #[test]
    fn accumulate_partitions_points() {
        // 4 points: two near (0,0), two near (10,10)
        let pts = vec![0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 9.0, 11.0];
        let cents = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let (sums, counts, inertia) = assign_accumulate(&pts, 4, 2, &cents);
        assert_eq!(counts, vec![2, 2]);
        // cluster 0 sum = (0+1, 0+0) = (1, 0)
        assert_eq!(sums[0], vec![1.0, 0.0]);
        // cluster 1 sum = (10+9, 10+11) = (19, 21)
        assert_eq!(sums[1], vec![19.0, 21.0]);
        assert!(inertia >= 0.0);
    }

    #[test]
    fn labels_match_assignment() {
        let pts = vec![0.0, 0.0, 10.0, 10.0];
        let cents = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        assert_eq!(assign_labels(&pts, 2, 2, &cents), vec![0, 1]);
    }

    #[test]
    fn lloyd_converges_two_clusters() {
        // synthetic: 100 points around (0,0) and 100 around (5,5); 2 iterations
        // should separate them starting from mediocre centroids.
        let mut pts = Vec::new();
        for i in 0..100 {
            pts.push(0.1 * (i % 5) as f32);
            pts.push(0.1 * (i % 3) as f32);
        }
        for i in 0..100 {
            pts.push(5.0 + 0.1 * (i % 5) as f32);
            pts.push(5.0 + 0.1 * (i % 3) as f32);
        }
        let n = 200;
        let mut cents = vec![vec![0.5, 0.5], vec![4.0, 4.0]];
        let mut last_inertia = f64::INFINITY;
        for _ in 0..5 {
            let (sums, counts, inertia) = assign_accumulate(&pts, n, 2, &cents);
            for c in 0..2 {
                if counts[c] > 0 {
                    for j in 0..2 {
                        cents[c][j] = (sums[c][j] / counts[c] as f64) as f32;
                    }
                }
            }
            assert!(inertia <= last_inertia + 1e-6, "inertia must not increase");
            last_inertia = inertia;
        }
        // centroids should land near (0,0) and (5,5)
        assert!(cents[0][0] < 1.0 && cents[1][0] > 4.0);
    }
}

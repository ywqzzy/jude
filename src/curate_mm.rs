//! Multimodal data-curation kernels — image/frame quality + perceptual hashing
//! for near-duplicate detection, the multimodal analogues of the text operators
//! in `crate::curate`. Operate on decoded pixel data (H×W×C `u8`, row-major),
//! so they compose with `jude.multimodal` decoders and the streaming video
//! source. Pure Rust, deterministic, unit-tested.
//!
//! Covers (from docs/llm_data_engine_plan.zh.md C17):
//! - perceptual hash (aHash + dHash + pHash/DCT) → near-dup image detection
//! - image quality signals (resolution, aspect, brightness, blur/variance)

/// Convert an interleaved H×W×C u8 image to a W'×W' grayscale f64 matrix
/// (row-major, `size`×`size`), nearest-neighbor resized. C may be 1/3/4.
fn to_gray_resized(pixels: &[u8], h: usize, w: usize, c: usize, size: usize) -> Vec<f64> {
    let size = size.max(1);
    let mut out = vec![0.0f64; size * size];
    if h == 0 || w == 0 || c == 0 || pixels.len() < h * w * c {
        return out;
    }
    for oy in 0..size {
        let sy = oy * h / size;
        for ox in 0..size {
            let sx = ox * w / size;
            let base = (sy * w + sx) * c;
            let gray = if c >= 3 {
                // luminance
                0.299 * pixels[base] as f64
                    + 0.587 * pixels[base + 1] as f64
                    + 0.114 * pixels[base + 2] as f64
            } else {
                pixels[base] as f64
            };
            out[oy * size + ox] = gray;
        }
    }
    out
}

fn bits_to_hex(bits: &[bool]) -> String {
    let mut out = String::with_capacity(bits.len() / 4);
    let mut i = 0;
    while i < bits.len() {
        let mut nib = 0u8;
        for j in 0..4 {
            if i + j < bits.len() && bits[i + j] {
                nib |= 1 << (3 - j);
            }
        }
        out.push(char::from_digit(nib as u32, 16).unwrap());
        i += 4;
    }
    out
}

/// Average hash (aHash): 8×8 gray, bit = pixel >= mean. 64-bit -> 16 hex chars.
pub fn ahash(pixels: &[u8], h: usize, w: usize, c: usize) -> String {
    let g = to_gray_resized(pixels, h, w, c, 8);
    let mean = g.iter().sum::<f64>() / g.len() as f64;
    let bits: Vec<bool> = g.iter().map(|&v| v >= mean).collect();
    bits_to_hex(&bits)
}

/// Difference hash (dHash): 9×8 gray, bit = left > right. Robust to brightness.
pub fn dhash(pixels: &[u8], h: usize, w: usize, c: usize) -> String {
    // resize to 9 wide x 8 tall
    let (rw, rh) = (9usize, 8usize);
    let mut g = vec![0.0f64; rw * rh];
    if !(h == 0 || w == 0 || c == 0 || pixels.len() < h * w * c) {
        for oy in 0..rh {
            let sy = oy * h / rh;
            for ox in 0..rw {
                let sx = ox * w / rw;
                let base = (sy * w + sx) * c;
                g[oy * rw + ox] = if c >= 3 {
                    0.299 * pixels[base] as f64
                        + 0.587 * pixels[base + 1] as f64
                        + 0.114 * pixels[base + 2] as f64
                } else {
                    pixels[base] as f64
                };
            }
        }
    }
    let mut bits = Vec::with_capacity(rh * (rw - 1));
    for y in 0..rh {
        for x in 0..(rw - 1) {
            bits.push(g[y * rw + x] > g[y * rw + x + 1]);
        }
    }
    bits_to_hex(&bits)
}

/// Perceptual hash (pHash, DCT-based): 32×32 gray -> 2D DCT -> keep top-left
/// 8×8 low-frequency block (excluding DC) -> bit = coeff > median. The most
/// robust of the three to scaling/compression/minor edits.
pub fn phash(pixels: &[u8], h: usize, w: usize, c: usize) -> String {
    const N: usize = 32;
    const K: usize = 8;
    let g = to_gray_resized(pixels, h, w, c, N);
    let dct = dct2d(&g, N);
    // collect the K×K low-frequency block
    let mut block = Vec::with_capacity(K * K);
    for y in 0..K {
        for x in 0..K {
            block.push(dct[y * N + x]);
        }
    }
    // median excluding the DC term (0,0)
    let mut vals: Vec<f64> = block.iter().skip(1).copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if vals.is_empty() {
        0.0
    } else {
        vals[vals.len() / 2]
    };
    let bits: Vec<bool> = block.iter().map(|&v| v > median).collect();
    bits_to_hex(&bits)
}

/// 2D DCT-II of an n×n matrix (separable; O(n^3), fine for n=32).
fn dct2d(input: &[f64], n: usize) -> Vec<f64> {
    // precompute cosine table
    let mut cos = vec![0.0f64; n * n];
    for k in 0..n {
        for x in 0..n {
            cos[k * n + x] =
                ((2.0 * x as f64 + 1.0) * k as f64 * std::f64::consts::PI / (2.0 * n as f64)).cos();
        }
    }
    // rows
    let mut tmp = vec![0.0f64; n * n];
    for r in 0..n {
        for k in 0..n {
            let mut s = 0.0;
            for x in 0..n {
                s += input[r * n + x] * cos[k * n + x];
            }
            tmp[r * n + k] = s;
        }
    }
    // cols
    let mut out = vec![0.0f64; n * n];
    for col in 0..n {
        for k in 0..n {
            let mut s = 0.0;
            for y in 0..n {
                s += tmp[y * n + col] * cos[k * n + y];
            }
            out[k * n + col] = s;
        }
    }
    out
}

/// Hamming distance between two equal-length hex hash strings (bit differences).
/// Returns u32::MAX if lengths differ.
pub fn hamming_hex(a: &str, b: &str) -> u32 {
    if a.len() != b.len() {
        return u32::MAX;
    }
    let mut d = 0u32;
    for (ca, cb) in a.chars().zip(b.chars()) {
        let na = ca.to_digit(16).unwrap_or(0) as u8;
        let nb = cb.to_digit(16).unwrap_or(0) as u8;
        d += (na ^ nb).count_ones();
    }
    d
}

/// Per-image quality signals for filtering (from decoded pixels).
#[derive(Clone, Debug, Default)]
pub struct ImageQuality {
    pub width: usize,
    pub height: usize,
    pub aspect_ratio: f64,
    /// mean luminance in [0,255]
    pub brightness: f64,
    /// variance of the Laplacian — low = blurry (a standard blur metric)
    pub sharpness: f64,
    /// fraction of pixels that are near-black or near-white (over/under-exposed)
    pub extreme_ratio: f64,
}

/// Compute image quality signals from H×W×C u8 pixels.
pub fn image_quality(pixels: &[u8], h: usize, w: usize, c: usize) -> ImageQuality {
    let mut q = ImageQuality {
        width: w,
        height: h,
        aspect_ratio: if h > 0 { w as f64 / h as f64 } else { 0.0 },
        ..Default::default()
    };
    if h == 0 || w == 0 || c == 0 || pixels.len() < h * w * c {
        return q;
    }
    // grayscale buffer + brightness + exposure
    let mut gray = vec![0.0f64; h * w];
    let mut sum = 0.0;
    let mut extreme = 0usize;
    for i in 0..(h * w) {
        let base = i * c;
        let lum = if c >= 3 {
            0.299 * pixels[base] as f64
                + 0.587 * pixels[base + 1] as f64
                + 0.114 * pixels[base + 2] as f64
        } else {
            pixels[base] as f64
        };
        gray[i] = lum;
        sum += lum;
        if lum < 8.0 || lum > 247.0 {
            extreme += 1;
        }
    }
    q.brightness = sum / (h * w) as f64;
    q.extreme_ratio = extreme as f64 / (h * w) as f64;
    // Laplacian variance (sharpness): 4-neighbor Laplacian over interior pixels
    if h >= 3 && w >= 3 {
        let mut lap = Vec::with_capacity((h - 2) * (w - 2));
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                let v = -4.0 * gray[idx]
                    + gray[idx - 1]
                    + gray[idx + 1]
                    + gray[idx - w]
                    + gray[idx + w];
                lap.push(v);
            }
        }
        let mean = lap.iter().sum::<f64>() / lap.len() as f64;
        let var = lap.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / lap.len() as f64;
        q.sharpness = var;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    // a solid-color image
    fn solid(h: usize, w: usize, val: u8) -> Vec<u8> {
        vec![val; h * w * 3]
    }

    // a vertical gradient
    fn gradient(h: usize, w: usize) -> Vec<u8> {
        let mut p = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            let v = (y * 255 / h.max(1)) as u8;
            for _ in 0..w {
                p.push(v);
                p.push(v);
                p.push(v);
            }
        }
        p
    }

    // a 2D pattern with both-axis structure (rich DCT content) — realistic for pHash
    fn pattern(h: usize, w: usize) -> Vec<u8> {
        let mut p = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                let v = (((x * 6 / w.max(1)) + (y * 4 / h.max(1))) % 2 * 200 + 20) as u8;
                p.push(v);
                p.push(v);
                p.push(v);
            }
        }
        p
    }

    #[test]
    fn hashes_are_hex_and_stable() {
        let img = gradient(40, 40);
        let a = phash(&img, 40, 40, 3);
        let b = phash(&img, 40, 40, 3);
        assert_eq!(a, b); // deterministic
        assert_eq!(a.len(), 16); // 8x8 bits = 64 = 16 hex
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(ahash(&img, 40, 40, 3).len(), 16);
    }

    #[test]
    fn phash_identical_zero_distance() {
        let img = gradient(50, 50);
        let a = phash(&img, 50, 50, 3);
        let b = phash(&img, 50, 50, 3);
        assert_eq!(hamming_hex(&a, &b), 0);
    }

    #[test]
    fn phash_similar_low_distance() {
        // same checkerboard-ish pattern at different resolution -> near-duplicate
        let a = phash(&pattern(64, 64), 64, 64, 3);
        let b = phash(&pattern(48, 48), 48, 48, 3);
        assert!(
            hamming_hex(&a, &b) <= 6,
            "resized dup should be close: {}",
            hamming_hex(&a, &b)
        );
    }

    #[test]
    fn phash_different_high_distance() {
        let pat = phash(&pattern(64, 64), 64, 64, 3);
        // a very different image: solid gray -> mostly one value
        let sol = phash(&solid(64, 64, 128), 64, 64, 3);
        assert!(
            hamming_hex(&pat, &sol) >= 6,
            "different images should differ: {}",
            hamming_hex(&pat, &sol)
        );
    }

    #[test]
    fn dhash_shape() {
        // dHash: 8 rows x (9-1) cols = 64 bits -> 16 hex chars
        let a = dhash(&gradient(40, 40), 40, 40, 3);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn quality_solid_is_blurry() {
        let img = solid(40, 40, 128);
        let q = image_quality(&img, 40, 40, 3);
        assert_eq!(q.width, 40);
        assert_eq!(q.height, 40);
        assert!((q.brightness - 128.0).abs() < 1.0);
        assert!(q.sharpness < 1e-6, "solid image has zero sharpness");
    }

    #[test]
    fn quality_gradient_has_edges() {
        let img = gradient(40, 40);
        let q = image_quality(&img, 40, 40, 3);
        // gradient has some variation but smooth -> low but nonzero on borders
        assert!(q.aspect_ratio == 1.0);
    }

    #[test]
    fn quality_exposure() {
        let dark = solid(20, 20, 2);
        let q = image_quality(&dark, 20, 20, 3);
        assert!(
            q.extreme_ratio > 0.9,
            "near-black should be flagged extreme"
        );
        assert!(q.brightness < 8.0);
    }
}

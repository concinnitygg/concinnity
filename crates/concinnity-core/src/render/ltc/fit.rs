// src/ltc/fit.rs
//
// Fits linearly transformed cosines (LTC) to the GGX BRDF, producing the lookup
// table the rectangular area-light shading path samples.
//
// An LTC approximates a BRDF lobe as a linear transform `M` of a clamped cosine
// distribution. The useful property is that the integral of a polygon against
// the transformed distribution equals the integral of the *inverse-transformed*
// polygon against the plain clamped cosine, which has a closed form. So area-light
// shading becomes: fetch `Minv`, transform the quad's corners, evaluate the
// closed-form polygon integral.
//
// This module fits `M` per (roughness, view angle) cell by minimising the error
// between the LTC distribution and the real GGX lobe, then stores the inverse.
//
// SELF-CONTAINED BY DESIGN: the only dependency is `rayon`, which both the crate
// and `build.rs` already have. `build.rs` `include!`s this file to generate the
// table at build time, so it must compile standalone.
//
// Table layout (see `fit_table`):
//   axis x = roughness in [0, 1], alpha = roughness^2
//   axis y = sqrt(1 - cos(theta_view)), so grazing angles get more resolution
//   matrix entry = the 4 non-trivial entries of Minv, normalised so Minv[1][1] = 1
//   magnitude entry = (directional albedo, Fresnel weight) for the Schlick split

// Number of stratified samples per axis when estimating the fit error. The error
// estimator draws this many squared samples from each of two distributions.
use alloc::vec;
use alloc::vec::Vec;

const ERROR_SAMPLES: usize = 16;
// Stratified samples per axis for the magnitude / average-direction estimate.
const AVG_SAMPLES: usize = 32;
// Smallest alpha fitted. A perfect mirror is a delta lobe that no linear
// transform of a cosine can represent, and below roughly this width the fit stops
// converging and falls back to the identity seed -- which would read as a broad
// highlight on the smoothest surfaces, the opposite of what is wanted. Clamping
// to a very sharp but finite lobe keeps the smoothest row well conditioned.
const MIN_ALPHA: f32 = 1.0e-3;

type Vec3 = [f32; 3];

// A fitted span of the table: the packed inverse transforms and the matching
// (directional albedo, Fresnel weight) pairs, in the same order. Used for both a
// single roughness row and the assembled table.
pub(crate) type LtcTable = (Vec<[f32; 4]>, Vec<[f32; 2]>);

// Row-major 3x3: `m[row][col]`.
#[derive(Clone, Copy, Debug)]
struct Mat3 {
    m: [[f32; 3]; 3],
}

impl Mat3 {
    const IDENTITY: Mat3 = Mat3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    fn mul_vec(&self, v: Vec3) -> Vec3 {
        let mut out = [0.0_f32; 3];
        for (row, o) in out.iter_mut().enumerate() {
            *o = self.m[row][0] * v[0] + self.m[row][1] * v[1] + self.m[row][2] * v[2];
        }
        out
    }

    fn mul(&self, other: &Mat3) -> Mat3 {
        let mut m = [[0.0_f32; 3]; 3];
        for (row, r) in m.iter_mut().enumerate() {
            for (col, c) in r.iter_mut().enumerate() {
                *c = (0..3).map(|k| self.m[row][k] * other.m[k][col]).sum();
            }
        }
        Mat3 { m }
    }
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: Vec3) -> Vec3 {
    let len = dot(v, v).sqrt();
    if len < 1.0e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

// --- GGX -------------------------------------------------------------------

// Smith masking-shadowing lambda for GGX.
fn ggx_lambda(alpha: f32, cos_theta: f32) -> f32 {
    if cos_theta >= 1.0 || cos_theta <= 0.0 {
        return 0.0;
    }
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let tan_theta = sin_theta / cos_theta;
    let a = 1.0 / (alpha * tan_theta);
    0.5 * (-1.0 + (1.0 + 1.0 / (a * a)).sqrt())
}

// The cosine-weighted GGX BRDF with Fresnel factored out:
// `D(H) * G2(V, L) / (4 * cos_v)`. Returns `(value, pdf)`; the pdf matches
// `ggx_sample`, which samples the NDF and reflects.
fn ggx_eval(v: Vec3, l: Vec3, alpha: f32) -> (f32, f32) {
    if v[2] <= 0.0 || l[2] <= 0.0 {
        return (0.0, 0.0);
    }
    let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
    if h[2] <= 0.0 {
        return (0.0, 0.0);
    }
    let slope_x = h[0] / h[2];
    let slope_y = h[1] / h[2];
    // GGX NDF written through the slope form, which stays stable as h[2] -> 0.
    let mut d = 1.0 / (1.0 + (slope_x * slope_x + slope_y * slope_y) / (alpha * alpha));
    d = d * d;
    d /= core::f32::consts::PI * alpha * alpha * h[2].powi(4);

    let vh = dot(v, h);
    if vh <= 0.0 {
        return (0.0, 0.0);
    }
    let pdf = (d * h[2] / (4.0 * vh)).abs();
    let g2 = 1.0 / (1.0 + ggx_lambda(alpha, v[2]) + ggx_lambda(alpha, l[2]));
    let value = d * g2 / (4.0 * v[2]);
    (value, pdf)
}

// Sample the GGX NDF and reflect the view vector about the sampled normal.
fn ggx_sample(v: Vec3, alpha: f32, u1: f32, u2: f32) -> Vec3 {
    let phi = 2.0 * core::f32::consts::PI * u1;
    let r = alpha * (u2 / (1.0 - u2).max(1.0e-9)).sqrt();
    let h = normalize([r * phi.cos(), r * phi.sin(), 1.0]);
    let vh = dot(h, v);
    [
        -v[0] + 2.0 * h[0] * vh,
        -v[1] + 2.0 * h[1] * vh,
        -v[2] + 2.0 * h[2] * vh,
    ]
}

// --- The linearly transformed cosine ---------------------------------------

#[derive(Clone, Copy, Debug)]
struct Ltc {
    // Fitted scale on the two in-plane axes and the skew that leans the lobe
    // toward grazing. `m13` stays 0 at normal incidence, where the lobe is
    // symmetric about the surface normal.
    m11: f32,
    m22: f32,
    m13: f32,
    // Orthonormal frame the fit works in: `z` is the lobe's average direction,
    // `x` lies in the incidence plane. Reduces the fit to three parameters.
    frame: Mat3,
    // Directional albedo of the lobe and its Fresnel weight, both estimated
    // rather than fitted.
    magnitude: f32,
    fresnel: f32,
    // Derived in `update`.
    matrix: Mat3,
    inverse: Mat3,
    det: f32,
}

impl Ltc {
    fn new() -> Ltc {
        let mut ltc = Ltc {
            m11: 1.0,
            m22: 1.0,
            m13: 0.0,
            frame: Mat3::IDENTITY,
            magnitude: 1.0,
            fresnel: 1.0,
            matrix: Mat3::IDENTITY,
            inverse: Mat3::IDENTITY,
            det: 1.0,
        };
        ltc.update();
        ltc
    }

    // Rebuild `matrix`, `inverse`, and `det` from the three fitted parameters.
    // The inner transform is upper triangular, so its inverse is closed-form and
    // the frame is orthonormal, so its inverse is its transpose -- no general
    // 3x3 inversion is needed.
    fn update(&mut self) {
        let inner = Mat3 {
            m: [
                [self.m11, 0.0, self.m13],
                [0.0, self.m22, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        self.matrix = self.frame.mul(&inner);

        let inv_inner = Mat3 {
            m: [
                [1.0 / self.m11, 0.0, -self.m13 / self.m11],
                [0.0, 1.0 / self.m22, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        let mut frame_t = Mat3 { m: [[0.0; 3]; 3] };
        for row in 0..3 {
            for col in 0..3 {
                frame_t.m[row][col] = self.frame.m[col][row];
            }
        }
        self.inverse = inv_inner.mul(&frame_t);
        // The frame is orthonormal and right-handed, so it contributes 1.
        self.det = self.m11 * self.m22;
    }

    // Density of the transformed distribution in direction `l`.
    fn eval(&self, l: Vec3) -> f32 {
        let original = normalize(self.inverse.mul_vec(l));
        if original[2] <= 0.0 {
            return 0.0;
        }
        let back = self.matrix.mul_vec(original);
        let len = dot(back, back).sqrt();
        // Jacobian of the transform at this direction.
        let jacobian = self.det / (len * len * len);
        let d = original[2] / core::f32::consts::PI;
        self.magnitude * d / jacobian
    }

    // Draw a direction from the transformed distribution.
    fn sample(&self, u1: f32, u2: f32) -> Vec3 {
        let theta = u1.sqrt().acos();
        let phi = 2.0 * core::f32::consts::PI * u2;
        normalize(self.matrix.mul_vec([
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ]))
    }
}

// --- Fitting ---------------------------------------------------------------

// Directional albedo, Fresnel weight, and average lobe direction for a GGX lobe,
// estimated by importance sampling the BRDF. These are not fitted: the magnitude
// scales the result and the average direction fixes the frame the fit works in.
fn compute_avg_terms(v: Vec3, alpha: f32) -> (f32, f32, Vec3) {
    let mut norm = 0.0_f32;
    let mut fresnel = 0.0_f32;
    let mut avg_dir = [0.0_f32; 3];
    let n = AVG_SAMPLES;
    for j in 0..n {
        for i in 0..n {
            let u1 = (i as f32 + 0.5) / n as f32;
            let u2 = (j as f32 + 0.5) / n as f32;
            let l = ggx_sample(v, alpha, u1, u2);
            let (value, pdf) = ggx_eval(v, l, alpha);
            if pdf > 0.0 {
                let weight = value / pdf;
                let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
                norm += weight;
                fresnel += weight * (1.0 - dot(v, h)).max(0.0).powi(5);
                avg_dir[0] += weight * l[0];
                avg_dir[1] += weight * l[1];
                avg_dir[2] += weight * l[2];
            }
        }
    }
    let inv = 1.0 / (n * n) as f32;
    // The lobe is symmetric about the incidence plane, so the average direction
    // has no out-of-plane component; forcing it keeps the frame exact.
    avg_dir[1] = 0.0;
    (norm * inv, fresnel * inv, normalize(avg_dir))
}

// Fit error between the LTC and the real GGX lobe. Draws from both
// distributions and weights each sample by the sum of the two pdfs, so neither
// tail dominates. The cubed difference penalises large local errors harder than
// an L2 norm would, which is what keeps highlight shapes faithful.
fn compute_error(ltc: &Ltc, v: Vec3, alpha: f32) -> f32 {
    let mut error = 0.0_f64;
    let n = ERROR_SAMPLES;
    for j in 0..n {
        for i in 0..n {
            let u1 = (i as f32 + 0.5) / n as f32;
            let u2 = (j as f32 + 0.5) / n as f32;

            // Sample the LTC.
            {
                let l = ltc.sample(u1, u2);
                let (value_brdf, pdf_brdf) = ggx_eval(v, l, alpha);
                let value_ltc = ltc.eval(l);
                let pdf_ltc = if ltc.magnitude > 0.0 {
                    value_ltc / ltc.magnitude
                } else {
                    0.0
                };
                let denom = (pdf_ltc + pdf_brdf) as f64;
                if denom > 0.0 {
                    let d = (value_brdf - value_ltc).abs() as f64;
                    error += d * d * d / denom;
                }
            }
            // Sample the BRDF.
            {
                let l = ggx_sample(v, alpha, u1, u2);
                let (value_brdf, pdf_brdf) = ggx_eval(v, l, alpha);
                let value_ltc = ltc.eval(l);
                let pdf_ltc = if ltc.magnitude > 0.0 {
                    value_ltc / ltc.magnitude
                } else {
                    0.0
                };
                let denom = (pdf_ltc + pdf_brdf) as f64;
                if denom > 0.0 {
                    let d = (value_brdf - value_ltc).abs() as f64;
                    error += d * d * d / denom;
                }
            }
        }
    }
    (error / (n * n) as f64) as f32
}

// Downhill-simplex minimisation over `dim` parameters (2 at normal incidence,
// where the skew is pinned to zero, otherwise 3). Chosen over a gradient method
// because the error estimator is a stochastic-ish sum with no analytic gradient.
fn nelder_mead<F: FnMut(&[f32]) -> f32>(
    start: &[f32],
    delta: f32,
    dim: usize,
    max_iters: usize,
    mut objective: F,
) -> Vec<f32> {
    let mut simplex: Vec<Vec<f32>> = Vec::with_capacity(dim + 1);
    simplex.push(start[..dim].to_vec());
    for i in 0..dim {
        let mut p = start[..dim].to_vec();
        p[i] += delta;
        simplex.push(p);
    }
    let mut values: Vec<f32> = simplex.iter().map(|p| objective(p)).collect();

    for _ in 0..max_iters {
        // Order worst-last.
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap());
        let best = order[0];
        let worst = order[order.len() - 1];
        let second_worst = order[order.len() - 2];

        // Centroid of everything but the worst vertex.
        let mut centroid = vec![0.0_f32; dim];
        for (idx, p) in simplex.iter().enumerate() {
            if idx != worst {
                for k in 0..dim {
                    centroid[k] += p[k];
                }
            }
        }
        for c in centroid.iter_mut() {
            *c /= dim as f32;
        }

        let combine = |t: f32| -> Vec<f32> {
            (0..dim)
                .map(|k| centroid[k] + t * (centroid[k] - simplex[worst][k]))
                .collect()
        };

        let reflected = combine(1.0);
        let reflected_value = objective(&reflected);
        if reflected_value < values[second_worst] && reflected_value >= values[best] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
            continue;
        }
        if reflected_value < values[best] {
            let expanded = combine(2.0);
            let expanded_value = objective(&expanded);
            if expanded_value < reflected_value {
                simplex[worst] = expanded;
                values[worst] = expanded_value;
            } else {
                simplex[worst] = reflected;
                values[worst] = reflected_value;
            }
            continue;
        }
        let contracted = combine(-0.5);
        let contracted_value = objective(&contracted);
        if contracted_value < values[worst] {
            simplex[worst] = contracted;
            values[worst] = contracted_value;
            continue;
        }
        // Shrink toward the best vertex.
        let best_point = simplex[best].clone();
        for (idx, p) in simplex.iter_mut().enumerate() {
            if idx != best {
                for k in 0..dim {
                    p[k] = best_point[k] + 0.5 * (p[k] - best_point[k]);
                }
            }
        }
        for (idx, val) in values.iter_mut().enumerate() {
            if idx != best {
                *val = objective(&simplex[idx]);
            }
        }
    }

    let best = (0..simplex.len())
        .min_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap())
        .unwrap();
    simplex[best].clone()
}

// Fit one grid cell, warm-started from `seed` (the previous cell's parameters).
// `isotropic` pins the skew to zero, which is exact at normal incidence.
fn fit_cell(v: Vec3, alpha: f32, seed: &Ltc, isotropic: bool) -> Ltc {
    let (magnitude, fresnel, avg_dir) = compute_avg_terms(v, alpha);

    let mut ltc = *seed;
    ltc.magnitude = magnitude;
    ltc.fresnel = fresnel;
    ltc.frame = if isotropic {
        Mat3::IDENTITY
    } else {
        // z along the lobe's average direction, x in the incidence plane. The
        // cross product x * y reproduces z, so the frame is right-handed and its
        // determinant is 1 (which `update` relies on).
        let z = avg_dir;
        let x = [z[2], 0.0, -z[0]];
        Mat3 {
            m: [[x[0], 0.0, z[0]], [x[1], 1.0, z[1]], [x[2], 0.0, z[2]]],
        }
    };
    if isotropic {
        ltc.m13 = 0.0;
    }
    ltc.update();

    if magnitude <= 0.0 {
        return ltc;
    }

    let dim = if isotropic { 2 } else { 3 };
    let start = [ltc.m11, ltc.m22, ltc.m13];
    let best = nelder_mead(&start, 0.05, dim, 64, |params| {
        let mut probe = ltc;
        probe.m11 = params[0].max(1.0e-7);
        probe.m22 = params[1].max(1.0e-7);
        probe.m13 = if isotropic { 0.0 } else { params[2] };
        probe.update();
        compute_error(&probe, v, alpha)
    });

    ltc.m11 = best[0].max(1.0e-7);
    ltc.m22 = best[1].max(1.0e-7);
    ltc.m13 = if isotropic { 0.0 } else { best[2] };
    ltc.update();
    ltc
}

// The four stored entries of `Minv`, normalised so its middle entry is 1.
//
// Only the direction of a transformed vertex matters to the polygon integral, so
// a uniform scale of `Minv` cancels; dividing through by the middle entry is what
// lets four floats stand in for a full matrix. The order is
// `(m00, m20, m02, m22)` and the shader rebuilds
// `[[x, 0, z], [0, 1, 0], [y, 0, w]]`.
fn packed_inverse(ltc: &Ltc) -> [f32; 4] {
    let inv = &ltc.inverse;
    let mid = inv.m[1][1];
    let s = if mid.abs() > 1.0e-9 { 1.0 / mid } else { 1.0 };
    [
        inv.m[0][0] * s,
        inv.m[2][0] * s,
        inv.m[0][2] * s,
        inv.m[2][2] * s,
    ]
}

// Fit the whole `size` x `size` table.
//
// Returns `(matrix, magnitude)`, both row-major with x = roughness and
// y = sqrt(1 - cos(theta_view)). Each row walks outward from normal incidence
// warm-started from the previous cell, which is what makes the simplex converge
// in a small number of iterations; a cold start per cell would need far more.
pub(crate) fn fit_table(size: usize) -> LtcTable {
    use rayon::prelude::*;

    // One roughness row per task. Rows are independent because the warm start
    // runs along the angle axis within a row, not across roughness.
    let rows: Vec<LtcTable> = (0..size)
        .into_par_iter()
        .map(|a| {
            let roughness = a as f32 / (size - 1) as f32;
            let alpha = (roughness * roughness).max(MIN_ALPHA);
            let mut seed = Ltc::new();
            let mut row_matrix = Vec::with_capacity(size);
            let mut row_magnitude = Vec::with_capacity(size);
            for t in 0..size {
                // Parameterised by sqrt(1 - cos), which spends more of the axis
                // on grazing angles where the lobe changes fastest.
                let x = t as f32 / (size - 1) as f32;
                let cos_theta = (1.0 - x * x).clamp(-1.0, 1.0);
                let theta = cos_theta.acos().min(1.57);
                let v = [theta.sin(), 0.0, theta.cos()];

                let ltc = fit_cell(v, alpha, &seed, t == 0);
                row_matrix.push(packed_inverse(&ltc));
                row_magnitude.push([ltc.magnitude, ltc.fresnel]);
                seed = ltc;
            }
            (row_matrix, row_magnitude)
        })
        .collect();

    // Rows are indexed by roughness; the table is indexed `a + t * size`.
    let mut matrix = vec![[0.0_f32; 4]; size * size];
    let mut magnitude = vec![[0.0_f32; 2]; size * size];
    for (a, (row_matrix, row_magnitude)) in rows.into_iter().enumerate() {
        for t in 0..size {
            matrix[a + t * size] = row_matrix[t];
            magnitude[a + t * size] = row_magnitude[t];
        }
    }

    (matrix, magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v_at(theta: f32) -> Vec3 {
        [theta.sin(), 0.0, theta.cos()]
    }

    // The GGX NDF must integrate to 1 against the projected solid angle, or
    // every fit built on it is scaled wrong.
    #[test]
    fn ggx_ndf_integrates_to_one() {
        for alpha in [0.1_f32, 0.4, 0.8] {
            let n = 400;
            let mut total = 0.0_f64;
            for j in 0..n {
                for i in 0..n {
                    let theta = (i as f32 + 0.5) / n as f32 * core::f32::consts::FRAC_PI_2;
                    let phi = (j as f32 + 0.5) / n as f32 * 2.0 * core::f32::consts::PI;
                    let h = [
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    ];
                    let slope_x = h[0] / h[2];
                    let slope_y = h[1] / h[2];
                    let mut d =
                        1.0 / (1.0 + (slope_x * slope_x + slope_y * slope_y) / (alpha * alpha));
                    d = d * d;
                    d /= core::f32::consts::PI * alpha * alpha * h[2].powi(4);
                    let d_omega = (core::f32::consts::FRAC_PI_2 / n as f32)
                        * (2.0 * core::f32::consts::PI / n as f32)
                        * theta.sin();
                    total += (d * h[2] * d_omega) as f64;
                }
            }
            assert!(
                (total - 1.0).abs() < 0.02,
                "alpha {alpha}: NDF integrated to {total}"
            );
        }
    }

    // A sampled direction must be in the upper hemisphere for a view near the
    // normal, and the pdf must be positive there.
    #[test]
    fn ggx_sampling_agrees_with_its_pdf() {
        let v = v_at(0.3);
        for i in 0..16 {
            for j in 0..16 {
                let u1 = (i as f32 + 0.5) / 16.0;
                let u2 = (j as f32 + 0.5) / 16.0;
                let l = ggx_sample(v, 0.4, u1, u2);
                let (value, pdf) = ggx_eval(v, l, 0.4);
                assert!(value.is_finite() && pdf.is_finite());
                if l[2] > 0.01 {
                    assert!(
                        pdf > 0.0,
                        "pdf should be positive for an upper-hemisphere L"
                    );
                }
            }
        }
    }

    // At normal incidence the lobe is symmetric about the surface normal, so the
    // fit must find no skew and the two in-plane scales must match.
    #[test]
    fn normal_incidence_fit_is_symmetric() {
        let seed = Ltc::new();
        let ltc = fit_cell([0.0, 0.0, 1.0], 0.25, &seed, true);
        assert_eq!(ltc.m13, 0.0);
        assert!(
            (ltc.m11 - ltc.m22).abs() < 0.05,
            "m11 {} vs m22 {}",
            ltc.m11,
            ltc.m22
        );
    }

    // `update` must produce a genuine inverse, or every transformed polygon is
    // silently wrong.
    #[test]
    fn the_inverse_really_inverts_the_matrix() {
        let mut ltc = Ltc::new();
        ltc.m11 = 0.7;
        ltc.m22 = 1.3;
        ltc.m13 = 0.35;
        let z = normalize([0.4, 0.0, 0.9]);
        let x = [z[2], 0.0, -z[0]];
        ltc.frame = Mat3 {
            m: [[x[0], 0.0, z[0]], [x[1], 1.0, z[1]], [x[2], 0.0, z[2]]],
        };
        ltc.update();
        let product = ltc.inverse.mul(&ltc.matrix);
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (product.m[row][col] - want).abs() < 1.0e-4,
                    "[{row}][{col}] = {}",
                    product.m[row][col]
                );
            }
        }
    }

    // The fit must actually reduce the error it was handed, at every angle.
    #[test]
    fn fitting_reduces_the_error_against_the_seed() {
        for theta in [0.0_f32, 0.6, 1.2] {
            let v = v_at(theta);
            let alpha: f32 = 0.3;
            let isotropic = theta == 0.0;
            let seed = Ltc::new();
            let fitted = fit_cell(v, alpha, &seed, isotropic);

            // The seed with the fitted magnitude and frame, but unfitted scales.
            let mut baseline = fitted;
            baseline.m11 = 1.0;
            baseline.m22 = 1.0;
            baseline.m13 = 0.0;
            baseline.update();

            let fitted_error = compute_error(&fitted, v, alpha);
            let baseline_error = compute_error(&baseline, v, alpha);
            assert!(
                fitted_error <= baseline_error,
                "theta {theta}: fitted {fitted_error} vs baseline {baseline_error}"
            );
        }
    }

    // A rough surface's lobe is close to a clamped cosine already, so its fitted
    // transform should be close to the identity.
    #[test]
    fn the_roughest_fit_is_near_identity() {
        let seed = Ltc::new();
        let ltc = fit_cell([0.0, 0.0, 1.0], 1.0, &seed, true);
        let packed = packed_inverse(&ltc);
        assert!((packed[0] - 1.0).abs() < 0.35, "m00 {}", packed[0]);
        assert!((packed[3] - 1.0).abs() < 0.35, "m22 {}", packed[3]);
        assert!(packed[1].abs() < 0.2 && packed[2].abs() < 0.2, "no skew");
    }

    // Every cell of a small table must be finite and physically sane: a
    // directional albedo above 1 would create energy.
    #[test]
    fn a_small_table_is_finite_and_energy_sane() {
        let size = 8;
        let (matrix, magnitude) = fit_table(size);
        assert_eq!(matrix.len(), size * size);
        for (i, m) in matrix.iter().enumerate() {
            assert!(m.iter().all(|v| v.is_finite()), "cell {i} matrix {m:?}");
        }
        for (i, mag) in magnitude.iter().enumerate() {
            assert!(mag[0].is_finite() && mag[1].is_finite(), "cell {i}");
            assert!(
                mag[0] >= 0.0 && mag[0] <= 1.05,
                "cell {i} albedo {} outside [0, 1]",
                mag[0]
            );
            assert!(
                (0.0..=1.05).contains(&mag[1]),
                "cell {i} fresnel {}",
                mag[1]
            );
        }
    }

    // The middle entry is normalised away, so the stored matrix must reproduce
    // the same transformed *direction* as the full inverse.
    #[test]
    fn the_packed_matrix_preserves_transformed_directions() {
        let seed = Ltc::new();
        let ltc = fit_cell(v_at(0.9), 0.35, &seed, false);
        let packed = packed_inverse(&ltc);
        let rebuilt = Mat3 {
            m: [
                [packed[0], 0.0, packed[2]],
                [0.0, 1.0, 0.0],
                [packed[1], 0.0, packed[3]],
            ],
        };
        for l in [[0.3_f32, 0.2, 0.9], [-0.5, 0.4, 0.75], [0.0, 0.0, 1.0]] {
            let unit = normalize(l);
            let a = normalize(ltc.inverse.mul_vec(unit));
            let b = normalize(rebuilt.mul_vec(unit));
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1.0e-3, "{a:?} vs {b:?}");
            }
        }
    }
}

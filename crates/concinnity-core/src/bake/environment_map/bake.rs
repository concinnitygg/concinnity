//! The IBL convolutions themselves: cube sampling, the two kernels, and the
//! row-sized unit of work they decompose into.
//!
//! Every output texel is an independent integral over a read-only source, so a
//! bake is expressed as [`CubeBake`] plus the rows of its output. The caller
//! decides whether to run those rows one at a time or all at once; this module
//! spawns nothing and takes no thread, so the two produce identical bytes.

use crate::math::{cos, floor, sin, sin_cos, sqrt};
use alloc::vec;
use alloc::vec::Vec;

use crate::gfx::cubemap;
use crate::math::vec3::{cross as cross3, dot as dot3, length};

/// Default azimuthal samples per irradiance texel.
pub const DEFAULT_IRRADIANCE_PHI_SAMPLES: u32 = 64;
/// Default polar samples per irradiance texel.
pub const DEFAULT_IRRADIANCE_THETA_SAMPLES: u32 = 16;

// Cube sampling

// Cube-face sampler: project a normalised direction onto the dominant axis
// to pick a face, then bilinearly sample within that face. Edges are clamped
// per-face (no seamless filtering).
fn sample_cube(faces: &[Vec<f32>; 6], face_size: u32, dir: [f32; 3]) -> [f32; 3] {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    let (face, ma, s, t) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 {
            (0usize, ax, -dir[2], -dir[1])
        } else {
            (1, ax, dir[2], -dir[1])
        }
    } else if ay >= az {
        if dir[1] > 0.0 {
            (2usize, ay, dir[0], dir[2])
        } else {
            (3, ay, dir[0], -dir[2])
        }
    } else if dir[2] > 0.0 {
        (4usize, az, dir[0], -dir[1])
    } else {
        (5, az, -dir[0], -dir[1])
    };
    let inv = 0.5 / ma.max(1e-20);
    let fs = face_size as f32;
    // s, t in [-1, 1] after multiplying by inv*2; map to pixel coords.
    let fx = (s * inv + 0.5) * fs - 0.5;
    let fy = (t * inv + 0.5) * fs - 0.5;
    let x0 = (floor(fx) as i32).clamp(0, face_size as i32 - 1);
    let y0 = (floor(fy) as i32).clamp(0, face_size as i32 - 1);
    let x1 = (x0 + 1).clamp(0, face_size as i32 - 1);
    let y1 = (y0 + 1).clamp(0, face_size as i32 - 1);
    let dx = (fx - floor(fx)).clamp(0.0, 1.0);
    let dy = (fy - floor(fy)).clamp(0.0, 1.0);
    let p = |x: i32, y: i32| -> [f32; 3] {
        let off = ((y as usize) * face_size as usize + x as usize) * 4;
        let face_data = &faces[face];
        [face_data[off], face_data[off + 1], face_data[off + 2]]
    };
    let p00 = p(x0, y0);
    let p10 = p(x1, y0);
    let p01 = p(x0, y1);
    let p11 = p(x1, y1);
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    [
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = length(v).max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}

// Build an orthonormal basis around `n` (N = up axis). Returns (tangent, bitangent).
fn make_tbn(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if n[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t = normalize3(cross3(up, n));
    let b = cross3(n, t);
    (t, b)
}

// Hammersley + GGX importance sampling

// Hammersley quasi-random 2D sequence over `n` samples. Used to drive the
// GGX importance sampler for prefilter convolution.
fn hammersley(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    let radical_inverse = (bits as f32) * 2.328_306_4e-10; // 1 / 2^32
    [i as f32 / n as f32, radical_inverse]
}

// Sample the GGX distribution in world space around normal `n`. Returns a
// half-vector H. The caller supplies the tangent basis and `a2m1` (a^2 - 1),
// both constant across a texel's whole sample set.
fn importance_sample_ggx(
    xi: [f32; 2],
    n: [f32; 3],
    basis: ([f32; 3], [f32; 3]),
    a2m1: f32,
) -> [f32; 3] {
    let (t, b) = basis;
    let phi = 2.0 * core::f32::consts::PI * xi[0];
    let cos_theta = sqrt((1.0 - xi[1]) / (1.0 + a2m1 * xi[1]));
    let sin_theta = sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    let (sin_phi, cos_phi) = sin_cos(phi);
    let h_local = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta];
    normalize3([
        t[0] * h_local[0] + b[0] * h_local[1] + n[0] * h_local[2],
        t[1] * h_local[0] + b[1] * h_local[1] + n[1] * h_local[2],
        t[2] * h_local[0] + b[2] * h_local[1] + n[2] * h_local[2],
    ])
}

// Cap a sampled radiance so a single very bright source texel (a sun disk, a
// blown sky highlight) cannot dominate a glossy reflection mip. Scales RGB
// uniformly to keep its hue when luminance exceeds `clamp`; `clamp <= 0`
// disables the cap. This suppresses the lone-hot-texel "bright squares" a
// clear-sky HDR otherwise smears across reflective floors. The rough mips always
// pass through here; mip 0 only does for a reflection probe (`clamp_mip0`), never
// for an imported environment map, so the on-screen skybox keeps its true HDR.
fn clamp_radiance(rgb: [f32; 3], clamp: f32) -> [f32; 3] {
    if clamp <= 0.0 {
        return rgb;
    }
    let lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if lum > clamp {
        let s = clamp / lum;
        return [rgb[0] * s, rgb[1] * s, rgb[2] * s];
    }
    rgb
}

// The chunked bake

/// One row of a bake's output: the RGBA texels of row `y` on cube face `face`.
///
/// Handed out by [`face_rows`] and filled by [`CubeBake::compute_row`]. A row
/// borrows only its own texels, so the whole set can be worked through in any
/// order and on any thread.
pub struct FaceRow<'a> {
    face: usize,
    y: u32,
    texels: &'a mut [f32],
}

/// Split six output faces of `face_size` edge into their rows, in face-major
/// order.
///
/// This is the unit of work an environment-map bake decomposes into: each row
/// reads only the immutable source and writes only the texels it holds, so a
/// caller is free to fan the set across a thread pool.
pub fn face_rows(faces: &mut [Vec<f32>; 6], face_size: u32) -> Vec<FaceRow<'_>> {
    let stride = face_size as usize * 4;
    faces
        .iter_mut()
        .enumerate()
        .flat_map(|(face, data)| {
            data.chunks_mut(stride)
                .enumerate()
                .map(move |(y, texels)| FaceRow {
                    face,
                    y: y as u32,
                    texels,
                })
        })
        .collect()
}

// The per-texel integral a bake evaluates. Both kernels precompute the terms
// that are constant across a whole bake so the inner loops do not.
enum Kernel {
    Irradiance(IrradianceKernel),
    Ggx(GgxKernel),
}

struct IrradianceKernel {
    phi_samples: u32,
    theta_samples: u32,
    inv_n_phi: f32,
    inv_n_theta: f32,
    weight: f32,
}

struct GgxKernel {
    samples: u32,
    a2m1: f32,
    clamp: f32,
}

/// A convolution of a source cubemap into a new cubemap, decomposed into
/// independent output rows.
///
/// [`compute`](Self::compute) runs the whole thing in one call; a caller that
/// wants the rows spread over worker threads takes [`output_faces`] and
/// [`face_rows`] and drives [`compute_row`] itself. Both produce the same
/// bytes: no row observes another.
///
/// [`output_faces`]: Self::output_faces
/// [`compute_row`]: Self::compute_row
pub struct CubeBake<'a> {
    source: &'a [Vec<f32>; 6],
    source_face_size: u32,
    output_face_size: u32,
    kernel: Kernel,
}

impl<'a> CubeBake<'a> {
    /// Cosine-weighted hemisphere integral over each output direction, by
    /// uniform (phi, theta) sampling. The result includes the cosine and
    /// Jacobian terms, so a shader plugs it straight in as
    /// `irradiance / π * albedo`.
    pub fn irradiance(
        source: &'a [Vec<f32>; 6],
        source_face_size: u32,
        output_face_size: u32,
        phi_samples: u32,
        theta_samples: u32,
    ) -> Self {
        let inv_n_phi = 1.0 / phi_samples as f32;
        let inv_n_theta = 1.0 / theta_samples as f32;
        // discrete weight: (Δθ * Δφ) = (π/2 / N_θ) * (2π / N_φ) = π² / (N_θ N_φ)
        let weight = core::f32::consts::PI * core::f32::consts::PI * inv_n_phi * inv_n_theta;
        Self {
            source,
            source_face_size,
            output_face_size,
            kernel: Kernel::Irradiance(IrradianceKernel {
                phi_samples,
                theta_samples,
                inv_n_phi,
                inv_n_theta,
                weight,
            }),
        }
    }

    /// GGX convolution at `roughness`, importance sampled with `samples`
    /// Hammersley points per texel and capped at `clamp` (see the firefly
    /// suppression in the module prose).
    pub fn ggx(
        source: &'a [Vec<f32>; 6],
        source_face_size: u32,
        output_face_size: u32,
        roughness: f32,
        samples: u32,
        clamp: f32,
    ) -> Self {
        let a = roughness * roughness;
        Self {
            source,
            source_face_size,
            output_face_size,
            kernel: Kernel::Ggx(GgxKernel {
                samples,
                a2m1: a * a - 1.0,
                clamp,
            }),
        }
    }

    /// Cube face edge of this bake's output, in pixels.
    pub fn output_face_size(&self) -> u32 {
        self.output_face_size
    }

    /// Six zeroed RGBA32F faces sized for this bake's output.
    pub fn output_faces(&self) -> [Vec<f32>; 6] {
        let f = self.output_face_size as usize;
        core::array::from_fn(|_| vec![0.0; f * f * 4])
    }

    /// Convolve the source into six output faces, handing the independent rows
    /// to `scheduler`. A caller with a thread pool gets the fan-out for free;
    /// one without uses [`Serial`](super::schedule::Serial).
    pub fn bake<S: super::schedule::RowScheduler>(&self, scheduler: &S) -> [Vec<f32>; 6] {
        let mut faces = self.output_faces();
        let mut rows = face_rows(&mut faces, self.output_face_size());
        scheduler.run(&mut rows, &|row| self.compute_row(row));
        faces
    }

    /// Evaluate one output row. Reads only the source, writes only `row`.
    pub fn compute_row(&self, row: &mut FaceRow<'_>) {
        match &self.kernel {
            Kernel::Irradiance(k) => self.irradiance_row(row, k),
            Kernel::Ggx(k) => self.ggx_row(row, k),
        }
    }

    /// Evaluate every row into a fresh output cube.
    pub fn compute(&self) -> [Vec<f32>; 6] {
        let mut faces = self.output_faces();
        for row in &mut face_rows(&mut faces, self.output_face_size) {
            self.compute_row(row);
        }
        faces
    }

    fn irradiance_row(&self, row: &mut FaceRow<'_>, k: &IrradianceKernel) {
        for x in 0..self.output_face_size {
            let n = cubemap::texel_dir(row.face, x, row.y, self.output_face_size);
            let (tan, bit) = make_tbn(n);
            let mut sum = [0.0f32; 3];
            for phi_i in 0..k.phi_samples {
                let phi = 2.0 * core::f32::consts::PI * (phi_i as f32 + 0.5) * k.inv_n_phi;
                let sin_phi = sin(phi);
                let cos_phi = cos(phi);
                for theta_i in 0..k.theta_samples {
                    let theta =
                        0.5 * core::f32::consts::PI * (theta_i as f32 + 0.5) * k.inv_n_theta;
                    let sin_theta = sin(theta);
                    let cos_theta = cos(theta);
                    let l_local = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta];
                    let dir = [
                        tan[0] * l_local[0] + bit[0] * l_local[1] + n[0] * l_local[2],
                        tan[1] * l_local[0] + bit[1] * l_local[1] + n[1] * l_local[2],
                        tan[2] * l_local[0] + bit[2] * l_local[1] + n[2] * l_local[2],
                    ];
                    let env = sample_cube(self.source, self.source_face_size, normalize3(dir));
                    // cos(θ) for the Lambert cosine, sin(θ) for the spherical
                    // area element. Both already in [0, 1] for the hemisphere.
                    let w = cos_theta * sin_theta;
                    sum[0] += env[0] * w;
                    sum[1] += env[1] * w;
                    sum[2] += env[2] * w;
                }
            }
            let off = x as usize * 4;
            row.texels[off] = sum[0] * k.weight;
            row.texels[off + 1] = sum[1] * k.weight;
            row.texels[off + 2] = sum[2] * k.weight;
            row.texels[off + 3] = 1.0;
        }
    }

    fn ggx_row(&self, row: &mut FaceRow<'_>, k: &GgxKernel) {
        for x in 0..self.output_face_size {
            let n = cubemap::texel_dir(row.face, x, row.y, self.output_face_size);
            // The tangent basis depends only on N, so it is built once per
            // texel rather than once per sample.
            let basis = make_tbn(n);
            // Split-sum approximation: V = R = N. The light direction is
            // then L = reflect(-V, H) = 2 (N·H) H - N.
            let mut accum = [0.0f32; 3];
            let mut total_weight = 0.0f32;
            for i in 0..k.samples {
                let xi = hammersley(i, k.samples);
                let h = importance_sample_ggx(xi, n, basis, k.a2m1);
                let ndh = dot3(n, h);
                if ndh <= 0.0 {
                    continue;
                }
                let l = normalize3([
                    2.0 * ndh * h[0] - n[0],
                    2.0 * ndh * h[1] - n[1],
                    2.0 * ndh * h[2] - n[2],
                ]);
                let ndl = dot3(n, l).max(0.0);
                if ndl > 0.0 {
                    let env =
                        clamp_radiance(sample_cube(self.source, self.source_face_size, l), k.clamp);
                    accum[0] += env[0] * ndl;
                    accum[1] += env[1] * ndl;
                    accum[2] += env[2] * ndl;
                    total_weight += ndl;
                }
            }
            let off = x as usize * 4;
            if total_weight > 0.0 {
                let inv = 1.0 / total_weight;
                row.texels[off] = accum[0] * inv;
                row.texels[off + 1] = accum[1] * inv;
                row.texels[off + 2] = accum[2] * inv;
            } else {
                let n_sample = sample_cube(self.source, self.source_face_size, n);
                row.texels[off] = n_sample[0];
                row.texels[off + 1] = n_sample[1];
                row.texels[off + 2] = n_sample[2];
            }
            row.texels[off + 3] = 1.0;
        }
    }
}

// Prefilter chain

/// Mip 0 of a prefilter chain: the source copied through at alpha 1.
///
/// `clamp_mip0` decides whether the firefly cap the rough mips always apply
/// also caps this mirror mip. An imported environment map leaves it OFF -- mip 0
/// is drawn directly as the on-screen skybox, which must keep its true HDR
/// sun/sky. A reflection probe turns it ON -- the probe is never a skybox (it is
/// sampled only by the specular term, as a low-res fallback when SSR/RT miss on
/// a near-mirror surface), so a lone blown highlight in the capture would
/// otherwise alias into a bright square there; capping it (at the same `clamp`)
/// suppresses that without touching any sky.
pub fn prefilter_mip0(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    clamp: f32,
    clamp_mip0: bool,
) -> [Vec<f32>; 6] {
    let f = source_face_size as usize;
    let mut mip0: [Vec<f32>; 6] = core::array::from_fn(|_| vec![0.0; f * f * 4]);
    for face in 0..6 {
        for i in 0..f * f {
            let off = i * 4;
            let mut rgb = [
                source[face][off],
                source[face][off + 1],
                source[face][off + 2],
            ];
            if clamp_mip0 {
                rgb = clamp_radiance(rgb, clamp);
            }
            mip0[face][off] = rgb[0];
            mip0[face][off + 1] = rgb[1];
            mip0[face][off + 2] = rgb[2];
            mip0[face][off + 3] = 1.0;
        }
    }
    mip0
}

/// GGX roughness for mip `mip` of a `mip_count` chain: 0 at mip 0, 1 at the
/// last mip.
pub fn prefilter_roughness(mip: u32, mip_count: u32) -> f32 {
    mip as f32 / (mip_count - 1) as f32
}

// Whole-bake entry points

/// Compute a low-resolution irradiance cubemap. The serial form of
/// [`CubeBake::irradiance`].
pub fn compute_irradiance(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    output_face_size: u32,
    phi_samples: u32,
    theta_samples: u32,
) -> [Vec<f32>; 6] {
    CubeBake::irradiance(
        source,
        source_face_size,
        output_face_size,
        phi_samples,
        theta_samples,
    )
    .compute()
}

/// Build a prefiltered radiance cube mip chain. Mip 0 is the unmodified
/// source (roughness=0 → Dirac lobe). Mip N is the GGX convolution at
/// roughness = N / (mip_count - 1). The serial form of [`prefilter_mip0`]
/// followed by one [`CubeBake::ggx`] per remaining mip.
pub fn compute_prefilter(
    source: &[Vec<f32>; 6],
    source_face_size: u32,
    mip_count: u32,
    samples_per_texel: u32,
    clamp: f32,
    clamp_mip0: bool,
) -> Vec<[Vec<f32>; 6]> {
    let mut mips: Vec<[Vec<f32>; 6]> = Vec::with_capacity(mip_count as usize);
    mips.push(prefilter_mip0(source, source_face_size, clamp, clamp_mip0));
    for mip in 1..mip_count {
        mips.push(
            CubeBake::ggx(
                source,
                source_face_size,
                source_face_size >> mip,
                prefilter_roughness(mip, mip_count),
                samples_per_texel,
                clamp,
            )
            .compute(),
        );
    }
    mips
}

#[cfg(test)]
mod tests {
    use super::super::schedule::{RowScheduler, Serial};
    use super::*;

    // A scheduler that walks the rows backwards. The bake's whole claim is that
    // the rows are independent, so the order it runs them in must not show up in
    // the output.
    struct Reversed;

    impl RowScheduler for Reversed {
        fn run<T: Send>(&self, items: &mut [T], compute: &(dyn Fn(&mut T) + Send + Sync)) {
            items.iter_mut().rev().for_each(compute);
        }
    }

    #[test]
    fn a_bake_does_not_depend_on_the_row_order() {
        let source = solid_cube(8, [0.4, 0.6, 0.9]);
        let bake = CubeBake::ggx(&source, 8, 8, 0.5, 16, 0.0);
        assert_eq!(bake.bake(&Serial), bake.bake(&Reversed));
    }

    #[test]
    fn a_bake_fills_every_output_face() {
        let source = solid_cube(8, [1.0, 1.0, 1.0]);
        let bake = CubeBake::irradiance(&source, 8, 4, 8, 8);
        let faces = bake.bake(&Serial);
        for face in &faces {
            assert_eq!(face.len(), 4 * 4 * 4);
            assert!(face.chunks_exact(4).all(|px| px[0] > 0.0));
        }
    }

    fn solid_cube(face_size: u32, color: [f32; 3]) -> [Vec<f32>; 6] {
        let f = face_size as usize;
        core::array::from_fn(|_| {
            let mut face = Vec::with_capacity(f * f * 4);
            for _ in 0..f * f {
                face.extend_from_slice(&[color[0], color[1], color[2], 1.0]);
            }
            face
        })
    }

    fn face_mean(face: &[f32]) -> [f32; 3] {
        let n = face.len() / 4;
        let mut m = [0.0f32; 3];
        for px in face.chunks_exact(4) {
            m[0] += px[0];
            m[1] += px[1];
            m[2] += px[2];
        }
        [m[0] / n as f32, m[1] / n as f32, m[2] / n as f32]
    }

    fn face_variance_red(face: &[f32]) -> f32 {
        let n = face.len() / 4;
        let mean = face.chunks_exact(4).map(|p| p[0]).sum::<f32>() / n as f32;

        face.chunks_exact(4)
            .map(|p| (p[0] - mean).powi(2))
            .sum::<f32>()
            / n as f32
    }

    // A source cube with one blazing texel on +Z over a uniform background: the
    // stand-in for a sun disk or a blown highlight in a probe capture.
    fn firefly_cube(face: usize) -> [Vec<f32>; 6] {
        let mut s: [Vec<f32>; 6] = core::array::from_fn(|_| vec![0.0; face * face * 4]);
        for fd in s.iter_mut() {
            for p in fd.chunks_exact_mut(4) {
                p[0] = 1.0;
                p[1] = 1.0;
                p[2] = 1.0;
                p[3] = 1.0;
            }
        }
        let off = ((face / 2) * face + face / 2) * 4;
        s[4][off] = 2000.0;
        s[4][off + 1] = 2000.0;
        s[4][off + 2] = 2000.0;
        s
    }

    fn peak(fd: &[f32]) -> f32 {
        fd.chunks_exact(4)
            .map(|p| p[0].max(p[1]).max(p[2]))
            .fold(0.0f32, f32::max)
    }

    // A source with structure in every direction, so a row that read the wrong
    // face or the wrong y would produce different bytes rather than the same
    // constant.
    fn gradient_cube(face_size: u32) -> [Vec<f32>; 6] {
        let f = face_size as usize;
        core::array::from_fn(|face| {
            let mut data = vec![0.0f32; f * f * 4];
            for y in 0..f {
                for x in 0..f {
                    let off = (y * f + x) * 4;
                    data[off] = face as f32 + x as f32 / f as f32;
                    data[off + 1] = y as f32 / f as f32;
                    data[off + 2] = (x + y) as f32 / (2 * f) as f32;
                    data[off + 3] = 1.0;
                }
            }
            data
        })
    }

    // The contract the chunk API exists for: a caller that drives the rows
    // itself, in an order nothing guarantees, gets the bytes `compute` would
    // have produced. Reversed here because a row that leaked state into the
    // next one would still pass in forward order.
    fn assert_rows_match_whole_image(bake: &CubeBake<'_>) {
        let whole = bake.compute();
        let mut chunked = bake.output_faces();
        let mut rows = face_rows(&mut chunked, bake.output_face_size());
        assert_eq!(
            rows.len(),
            6 * bake.output_face_size() as usize,
            "one row per face line"
        );
        for row in rows.iter_mut().rev() {
            bake.compute_row(row);
        }
        assert_eq!(chunked, whole, "chunked bake diverged from the whole image");
    }

    #[test]
    fn chunked_irradiance_matches_the_whole_image() {
        let source = gradient_cube(8);
        assert_rows_match_whole_image(&CubeBake::irradiance(&source, 8, 4, 16, 8));
    }

    #[test]
    fn chunked_ggx_matches_the_whole_image() {
        let source = gradient_cube(16);
        for roughness in [0.25f32, 0.5, 1.0] {
            assert_rows_match_whole_image(&CubeBake::ggx(&source, 16, 8, roughness, 32, 0.0));
        }
    }

    // The firefly cap is part of the kernel, so it has to survive the split too.
    #[test]
    fn chunked_ggx_matches_the_whole_image_under_the_firefly_cap() {
        let source = firefly_cube(16);
        assert_rows_match_whole_image(&CubeBake::ggx(&source, 16, 8, 0.5, 64, 8.0));
    }

    #[test]
    fn face_rows_cover_every_texel_exactly_once() {
        let mut faces: [Vec<f32>; 6] = core::array::from_fn(|_| vec![0.0; 4 * 4 * 4]);
        for row in &mut face_rows(&mut faces, 4) {
            assert_eq!(row.texels.len(), 4 * 4, "a row is one line of RGBA texels");
            for t in row.texels.iter_mut() {
                *t += 1.0;
            }
        }
        assert!(
            faces.iter().flatten().all(|&t| t == 1.0),
            "every texel written exactly once"
        );
    }

    #[test]
    fn hammersley_first_sample_is_zero() {
        let s = hammersley(0, 1024);
        assert!(s[0].abs() < 1e-6, "x was {}", s[0]);
        assert!(s[1].abs() < 1e-6, "y was {}", s[1]);
    }

    #[test]
    fn hammersley_last_sample_is_just_under_one() {
        let s = hammersley(1023, 1024);
        assert!(s[0] > 0.99 && s[0] < 1.0, "x was {}", s[0]);
    }

    #[test]
    fn importance_sample_ggx_at_xi_zero_returns_n() {
        let n = [0.0, 0.0, 1.0];
        let a = 0.5f32 * 0.5;
        let h = importance_sample_ggx([0.0, 0.0], n, make_tbn(n), a * a - 1.0);
        // xi=(0,0) → cos_theta = 1 → H aligns with N.
        assert!((h[0] - 0.0).abs() < 1e-5);
        assert!((h[1] - 0.0).abs() < 1e-5);
        assert!((h[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn irradiance_solid_color_is_pi_times_color() {
        // Uniform environment L = (1, 0.5, 0.25). The hemispherical integral
        // of L * cos(θ) over the upper hemisphere is π * L. The discrete
        // (phi, theta) integration should converge to that.
        let source = solid_cube(8, [1.0, 0.5, 0.25]);
        let irr = compute_irradiance(&source, 8, 4, 64, 16);
        let mean = face_mean(&irr[0]);
        let expected = [
            core::f32::consts::PI * 1.0,
            core::f32::consts::PI * 0.5,
            core::f32::consts::PI * 0.25,
        ];
        // Discrete integration loses a few percent; accept ±5%.
        for c in 0..3 {
            let delta = (mean[c] - expected[c]).abs() / expected[c];
            assert!(
                delta < 0.05,
                "channel {} mean {} expected {}",
                c,
                mean[c],
                expected[c]
            );
        }
    }

    #[test]
    fn prefilter_mip_zero_matches_source_with_alpha_one() {
        let source = solid_cube(16, [0.7, 0.3, 0.1]);
        let mips = compute_prefilter(&source, 16, 3, 16, 0.0, false);
        for face in &mips[0] {
            for px in 0..16 * 16 {
                let off = px * 4;
                assert!((face[off] - 0.7).abs() < 1e-6);
                assert!((face[off + 1] - 0.3).abs() < 1e-6);
                assert!((face[off + 2] - 0.1).abs() < 1e-6);
                assert!((face[off + 3] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn prefilter_solid_color_stays_solid_at_high_roughness() {
        let source = solid_cube(16, [0.5, 0.5, 0.5]);
        let mips = compute_prefilter(&source, 16, 4, 32, 0.0, false);
        // Last mip should still be ~0.5 grey since input is uniform.
        let mean = face_mean(&mips[3][0]);
        for (c, m) in mean.iter().enumerate() {
            assert!((m - 0.5).abs() < 0.02, "channel {} mean {}", c, m);
        }
    }

    #[test]
    fn prefilter_roughness_spans_zero_to_one() {
        assert_eq!(prefilter_roughness(0, 5), 0.0);
        assert_eq!(prefilter_roughness(4, 5), 1.0);
        assert_eq!(prefilter_roughness(2, 5), 0.5);
    }

    #[test]
    fn prefilter_blurs_a_red_seam() {
        // Place a bright red column on +Z face only; prefilter at roughness=1
        // should spread it across the face so the variance drops vs the input.
        let face = 16usize;
        let mut source: [Vec<f32>; 6] = core::array::from_fn(|_| vec![0.0; face * face * 4]);
        for face_data in source.iter_mut() {
            for p in face_data.chunks_exact_mut(4) {
                p[3] = 1.0;
            }
        }
        // +Z face (index 4): paint x=8 column bright red.
        for y in 0..face {
            let off = (y * face + 8) * 4;
            source[4][off] = 20.0;
        }
        let mips = compute_prefilter(&source, face as u32, 3, 256, 0.0, false);
        // Compare variance of +Z face at mip 0 vs mip 2.
        let v0 = face_variance_red(&mips[0][4]);
        let v2 = face_variance_red(&mips[2][4]);
        assert!(
            v2 < v0 * 0.5,
            "prefilter did not blur: mip 0 var={}, mip 2 var={}",
            v0,
            v2
        );
    }

    #[test]
    fn clamp_radiance_caps_luminance_and_keeps_hue() {
        // Below the cap: untouched.
        let dim = [1.0, 0.5, 0.25];
        assert_eq!(clamp_radiance(dim, 10.0), dim);
        // Disabled (clamp <= 0): untouched even when very bright.
        let hot = [100.0, 50.0, 25.0];
        assert_eq!(clamp_radiance(hot, 0.0), hot);
        // Above the cap: luminance is pulled to the cap, hue preserved.
        let capped = clamp_radiance(hot, 10.0);
        let lum = 0.2126 * capped[0] + 0.7152 * capped[1] + 0.0722 * capped[2];
        assert!((lum - 10.0).abs() < 1e-3, "luminance {} != cap 10", lum);
        assert!((capped[0] / capped[1] - hot[0] / hot[1]).abs() < 1e-4);
        assert!((capped[1] / capped[2] - hot[1] / hot[2]).abs() < 1e-4);
    }

    #[test]
    fn prefilter_clamp_suppresses_a_firefly() {
        // Unclamped the blazing texel survives the GGX convolution as a hot spot
        // that smears into bright squares; the cap spreads its energy so the
        // brightest reflection texel is far dimmer, while the background (below
        // the cap) is preserved.
        let face = 16usize;
        let unclamped = compute_prefilter(&firefly_cube(face), face as u32, 3, 256, 0.0, false);
        let clamped = compute_prefilter(&firefly_cube(face), face as u32, 3, 256, 8.0, false);
        let p_unclamped = peak(&unclamped[1][4]);
        let p_clamped = peak(&clamped[1][4]);
        assert!(
            p_clamped < p_unclamped * 0.5,
            "clamp did not suppress the firefly: unclamped {}, clamped {}",
            p_unclamped,
            p_clamped
        );
        assert!(
            p_clamped >= 0.9,
            "clamp crushed the background: clamped peak {}",
            p_clamped
        );
    }

    #[test]
    fn prefilter_mip0_clamp_caps_a_mirror_firefly_only_when_requested() {
        // Mip 0 is the mirror (roughness 0) reflection a near-mirror surface
        // samples on an SSR/RT miss.
        let face = 16usize;
        // clamp_mip0 = false (an imported env map / skybox): the blazing texel survives
        // mip 0 untouched, so the on-screen sky would keep its true HDR sun.
        let unclamped_mip0 = compute_prefilter(&firefly_cube(face), face as u32, 2, 16, 8.0, false);
        assert!(
            peak(&unclamped_mip0[0][4]) > 1000.0,
            "mip 0 should be unclamped when clamp_mip0 is false: peak {}",
            peak(&unclamped_mip0[0][4])
        );
        // clamp_mip0 = true (a reflection probe): the same firefly is capped at mip 0,
        // while the uniform background (below the cap) is preserved.
        let clamped_mip0 = compute_prefilter(&firefly_cube(face), face as u32, 2, 16, 8.0, true);
        let p = peak(&clamped_mip0[0][4]);
        assert!(p <= 8.0 + 1e-3, "mip 0 firefly not capped: peak {p}");
        assert!(p >= 0.9, "mip 0 clamp crushed the background: peak {p}");
    }
}

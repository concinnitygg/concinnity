//! The linearly-transformed-cosine lookup tables the rectangular area-light
//! shading path samples, generated at build time by the fitter in `fit.rs`.
//!
//! Two tables, both `LTC_LUT_SIZE` square and indexed the same way:
//!   u = roughness in [0, 1]
//!   v = sqrt(1 - cos(theta_view)), which spends more of the axis on the grazing
//!       angles where the lobe changes fastest
//!
//! `matrix_texels` holds 4 floats per cell: the non-trivial entries of the inverse
//! transform, normalised so the middle entry is 1. The shader rebuilds
//! `[[x, 0, z], [0, 1, 0], [y, 0, w]]`, transforms the light quad's corners by it,
//! and evaluates the closed-form clamped-cosine polygon integral.
//!
//! `magnitude_texels` holds 2 floats per cell: the lobe's directional albedo and
//! its Fresnel weight, recombined by the shader as
//! `f0 * albedo + (1 - f0) * fresnel`.

// The fitter runs from build.rs, which `include!`s it next to `size.rs`. The lib
// needs only the table size, so it compiles the fitter for its own tests alone.
#[cfg(test)]
mod fit;
// The CPU twin of the shader's polygon integral, kept so the closed form can
// be checked against brute-force Monte Carlo. Nothing else calls it.
#[cfg(test)]
mod polygon;
mod size;

pub use size::LTC_LUT_SIZE;

// build.rs emits raw little-endian f32 rather than Rust source, because a static
// array of this many float literals costs rustc tens of seconds to compile.
// Every target the engine builds for is little-endian.
#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

// Aligning the bytes to f32 is what lets the tables be read where they already
// are, with no decode pass and no copy on the heap.
static MATRIX: &Aligned<[u8]> =
    &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/ltc_matrix.bin")));
static MAGNITUDE: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/ltc_magnitude.bin"
)));

/// RGBA32Float texels, `LTC_LUT_SIZE` square. The backend uploads these once.
pub fn matrix_texels() -> &'static [f32] {
    bytemuck::cast_slice(&MATRIX.0)
}

/// RG32Float texels, `LTC_LUT_SIZE` square.
pub fn magnitude_texels() -> &'static [f32] {
    bytemuck::cast_slice(&MAGNITUDE.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generated_tables_are_the_expected_size() {
        assert_eq!(matrix_texels().len(), LTC_LUT_SIZE * LTC_LUT_SIZE * 4);
        assert_eq!(magnitude_texels().len(), LTC_LUT_SIZE * LTC_LUT_SIZE * 2);
    }

    // A NaN or infinity anywhere in the table would blow out every area-light
    // highlight that samples that cell.
    #[test]
    fn the_generated_tables_are_finite() {
        assert!(matrix_texels().iter().all(|v| v.is_finite()));
        assert!(magnitude_texels().iter().all(|v| v.is_finite()));
    }

    // The directional albedo is an energy fraction; above 1 it would create light.
    #[test]
    fn the_generated_albedo_conserves_energy() {
        for (i, chunk) in magnitude_texels().chunks_exact(2).enumerate() {
            assert!(
                (0.0..=1.05).contains(&chunk[0]),
                "cell {i} albedo {}",
                chunk[0]
            );
            assert!(
                (0.0..=1.05).contains(&chunk[1]),
                "cell {i} fresnel {}",
                chunk[1]
            );
        }
    }

    // The roughest, most head-on cell is where the GGX lobe is closest to a plain
    // clamped cosine, so its transform must come out near the identity. This is
    // the cheapest end-to-end check that the generated table is the fitter's
    // output and not stale or byte-swapped.
    #[test]
    fn the_roughest_head_on_cell_is_near_identity() {
        let m = matrix_texels();
        // Roughness 1 sits at the end of the first (head-on) row.
        let base = (LTC_LUT_SIZE - 1) * 4;
        assert!((m[base] - 1.0).abs() < 0.35, "m00 {}", m[base]);
        assert!((m[base + 3] - 1.0).abs() < 0.35, "m22 {}", m[base + 3]);
        assert!(
            m[base + 1].abs() < 0.2 && m[base + 2].abs() < 0.2,
            "no skew"
        );
    }

    // The lobe widens with roughness, so the transform's scale must grow along the
    // head-on roughness axis. This is the strongest single check that the fit
    // converged: a table that fell back to its identity seed would be flat here.
    //
    // The scale lives in entry 3, not entry 0. At normal incidence the two
    // in-plane axes are equal by isotropy, and entry 0 is their ratio, so it is
    // legitimately 1.0 at every roughness.
    #[test]
    fn the_transform_scale_grows_with_roughness_head_on() {
        let m = matrix_texels();
        let scale = |a: usize| m[a * 4 + 3];
        assert!(
            scale(0) < 0.05,
            "the smoothest surface should have a narrow lobe, got {}",
            scale(0)
        );
        assert!(
            scale(LTC_LUT_SIZE - 1) > 0.8,
            "the roughest surface should be near a cosine lobe, got {}",
            scale(LTC_LUT_SIZE - 1)
        );
        for a in 1..LTC_LUT_SIZE {
            assert!(
                scale(a) >= scale(a - 1) - 1.0e-3,
                "scale dipped at roughness index {a}: {} then {}",
                scale(a - 1),
                scale(a)
            );
        }
    }

    // At normal incidence the lobe is symmetric about the surface normal, so the
    // whole head-on row must be skew-free whatever the roughness.
    #[test]
    fn the_head_on_row_has_no_skew() {
        let m = matrix_texels();
        for a in 0..LTC_LUT_SIZE {
            assert!(
                m[a * 4 + 1].abs() < 1.0e-3 && m[a * 4 + 2].abs() < 1.0e-3,
                "roughness index {a} skewed at normal incidence"
            );
        }
    }
}

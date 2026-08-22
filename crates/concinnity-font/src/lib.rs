//! Build-time font compilation: rasterises the printable ASCII glyphs of a TTF
//! face with fontdue, packs them into a power-of-two RGBA atlas as a signed
//! distance field, and encodes the result as the blob payload
//! `concinnity_cpu::build::font::deserialise` reads back.
//!
//! Split out of the cook pipeline so both the asset build and the engine's own
//! build script can bake an atlas: the engine embeds one for its startup error
//! screen, which must render without any compiled world data.

mod payload;
mod sdf;

use concinnity_cpu::build::font::GlyphMetrics;
use sdf::{EdtScratch, SdfScratch, cell_coverage_to_sdf};

// Pixels of distance gradient on each side of the glyph edge stored in the atlas
// (in low-resolution / atlas pixels).
const SDF_SPREAD: f32 = 4.0;

// Glyphs are rasterised at this multiple of the requested size, the SDF is
// computed at that high resolution, and then box-filtered down to the atlas
// resolution. Oversampling avoids the staircase artefacts that come from
// thresholding a low-resolution coverage bitmap; the box filter averages
// OVERSAMPLE² high-res samples per atlas texel, so curves stay smooth.
const OVERSAMPLE: u32 = 8;

// Atlas supersampling: the final atlas stores each glyph at this multiple of the
// requested size, while the positional metrics (advance, bearing) stay in
// requested-size units, so on-screen layout is unchanged but the atlas carries
// SUPERSAMPLE times more texels per displayed pixel. HUD chips draw the font
// minified (small scale), so a 1x atlas undersamples and aliases; the extra
// texels let the renderer's trilinear mip chain supersample the glyph down
// cleanly. It also shrinks the SDF spread in screen terms (SDF_SPREAD is fixed
// in atlas texels), so thin strokes hold their contrast at small sizes instead
// of fading out of the antialiasing band.
const SUPERSAMPLE: u32 = 2;

// If 8× still isn't enough, the next lever isn't more oversample
// (diminishing returns and atlas memory growth), but switching the EDT to
// consume fontdue's antialiased coverage values directly instead of
// binary-thresholding them. That's the Gustavson 2012 anti-aliased EDT, and
// it places the implicit surface at sub-pixel positions derived from coverage.

/// Source filename of the bundled default face. Companion injection derives the
/// auto-injected Font asset's name from it, so a generated default font is named
/// exactly as `cn add` would name the same file.
pub const BUILTIN_FONT_FILE: &str = "Questrial-Regular.ttf";

/// The bundled default face, shipped in the binary so no external file is
/// required to compile an atlas.
pub const BUILTIN_FONT_BYTES: &[u8] = include_bytes!("fonts/Questrial-Regular.ttf");

/// Rasterise `ttf_bytes` at `size_px` and encode the atlas as a blob payload.
/// `source` names the face in error messages.
pub fn compile(ttf_bytes: &[u8], size_px: u32, source: &str) -> Result<Vec<u8>, String> {
    let logical_size_px = size_px as f32;
    // The whole pipeline (rasterise, SDF, pack) runs at the supersampled size, so
    // the atlas and its texel-space metrics come out SUPERSAMPLE times larger.
    // Positional metrics are divided back to `logical_size_px` units at emit time.
    let raster_size_px = logical_size_px * SUPERSAMPLE as f32;

    let settings = fontdue::FontSettings {
        scale: raster_size_px * OVERSAMPLE as f32,
        ..Default::default()
    };
    let font = fontdue::Font::from_bytes(ttf_bytes, settings)
        .map_err(|e| format!("Font: failed to parse '{}': {}", source, e))?;

    // Rasterise every printable ASCII character (32-126) at OVERSAMPLE × the
    // target size. The SDF is computed at this high resolution and box-filtered
    // back down to atlas resolution; the resulting low-res field captures
    // sub-pixel edge positions that a same-resolution threshold would lose.
    let chars: Vec<char> = (32u8..=126u8).map(|b| b as char).collect();
    let rast_size = raster_size_px * OVERSAMPLE as f32;

    let mut bitmaps: Vec<(char, Vec<u8>, fontdue::Metrics)> = Vec::new();
    for &ch in &chars {
        let (metrics, bitmap) = font.rasterize(ch, rast_size);
        bitmaps.push((ch, bitmap, metrics));
    }

    // Atlas layout is planned in low-res (final) pixels and scaled up by
    // OVERSAMPLE for the working high-res atlas, so atlas dimensions and
    // every glyph position downsample cleanly to integer low-res coordinates.
    const PAD_LO: u16 = 4;
    let pad_hi = PAD_LO * OVERSAMPLE as u16;
    let oversample_u16 = OVERSAMPLE as u16;

    // Per-glyph high-res sizes, rounded up to a multiple of OVERSAMPLE so each
    // glyph cell aligns to the low-res grid.
    let glyph_dims_hi: Vec<(u16, u16)> = bitmaps
        .iter()
        .map(|(_, _, m)| {
            (
                round_up_to(m.width as u16, oversample_u16),
                round_up_to(m.height as u16, oversample_u16),
            )
        })
        .collect();

    let max_glyph_w_hi = glyph_dims_hi.iter().map(|(w, _)| *w).max().unwrap_or(0) + pad_hi * 2;
    let max_glyph_h_hi = glyph_dims_hi.iter().map(|(_, h)| *h).max().unwrap_or(0) + pad_hi * 2;

    let glyph_count = bitmaps.len() as u16;
    // Compute the atlas layout in u32: at larger font sizes the high-res glyph
    // stride times the glyph count overflows u16 (a debug-only multiply panic),
    // even though the packed atlas width is then clamped to <= 2048 logical px.
    let ideal_w_hi = (max_glyph_w_hi as u32 + pad_hi as u32) * glyph_count as u32;
    let atlas_w_hi = u32::next_power_of_two(ideal_w_hi.max(64)).min(2048 * OVERSAMPLE) as u16;
    let glyphs_per_row = (atlas_w_hi / (max_glyph_w_hi + pad_hi)).max(1);
    let rows = glyph_count.div_ceil(glyphs_per_row);
    let atlas_h_hi = u32::next_power_of_two(
        (max_glyph_h_hi as u32 + pad_hi as u32) * rows as u32 + pad_hi as u32,
    ) as u16;

    let atlas_w_hi = atlas_w_hi as u32;
    let atlas_h_hi = atlas_h_hi as u32;
    debug_assert_eq!(atlas_w_hi % OVERSAMPLE, 0);
    debug_assert_eq!(atlas_h_hi % OVERSAMPLE, 0);

    // Each glyph is processed in its own cell buffer rather than a shared
    // high-res atlas. This keeps the EDT and box-filter working on a small
    // region (~cell_w×cell_h pixels) instead of the full atlas (~33M pixels),
    // which makes a large difference in unoptimised (debug) builds.
    let cell_w_hi = max_glyph_w_hi as u32; // includes 2×pad_hi on each axis
    let cell_h_hi = max_glyph_h_hi as u32;
    let cell_w_lo = cell_w_hi / OVERSAMPLE;
    let cell_h_lo = cell_h_hi / OVERSAMPLE;
    let cell_n = (cell_w_hi * cell_h_hi) as usize;
    let block_count = OVERSAMPLE * OVERSAMPLE;
    let oversample_f = OVERSAMPLE as f32;
    let sdf_spread = SDF_SPREAD * oversample_f;

    // Final low-res atlas.
    let atlas_w = atlas_w_hi / OVERSAMPLE;
    let atlas_h = atlas_h_hi / OVERSAMPLE;
    let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];

    // Reusable per-glyph buffers, allocated once, cleared each iteration.
    let mut cell_hi = vec![0u8; cell_n * 4];

    // SDF scratch: per-pixel distance grids plus the EDT working buffers sized
    // for the largest cell dimension; reused across every glyph.
    let max_cell_dim = cell_w_hi.max(cell_h_hi) as usize;
    let mut sdf_scratch = SdfScratch {
        inside_dist2: vec![0.0f32; cell_n],
        outside_dist2: vec![0.0f32; cell_n],
        edt: EdtScratch {
            v: vec![0usize; max_cell_dim],
            z: vec![0.0f32; max_cell_dim + 1],
            row_tmp: vec![0.0f32; cell_w_hi as usize],
            col_src: vec![0.0f32; cell_h_hi as usize],
            col_dst: vec![0.0f32; cell_h_hi as usize],
        },
    };

    let mut metrics_out: Vec<GlyphMetrics> = Vec::new();

    for (i, (ch, bitmap, metrics)) in bitmaps.iter().enumerate() {
        let col = (i as u16) % glyphs_per_row;
        let row = (i as u16) / glyphs_per_row;
        let ax_hi = pad_hi + col * (max_glyph_w_hi + pad_hi);
        let ay_hi = pad_hi + row * (max_glyph_h_hi + pad_hi);

        let gw_raw = metrics.width as u16;
        let gh_raw = metrics.height as u16;

        // Fill cell_hi: zero it, then place glyph coverage at (pad_hi, pad_hi).
        cell_hi.fill(0);
        for py in 0..gh_raw {
            for px in 0..gw_raw {
                let src = (py as usize) * (gw_raw as usize) + px as usize;
                let dst = ((pad_hi as u32 + py as u32) * cell_w_hi + pad_hi as u32 + px as u32)
                    as usize
                    * 4;
                cell_hi[dst] = bitmap[src];
            }
        }

        // Compute per-glyph SDF using pre-allocated scratch.
        cell_coverage_to_sdf(
            &mut cell_hi,
            cell_w_hi as usize,
            cell_h_hi as usize,
            sdf_spread,
            &mut sdf_scratch,
        );

        // Box-filter the high-res cell into the final low-res atlas.
        // Cell origin in the low-res atlas matches the original layout.
        let cell_ax_lo = col as u32 * (cell_w_lo + PAD_LO as u32);
        let cell_ay_lo = row as u32 * (cell_h_lo + PAD_LO as u32);
        for ly in 0..cell_h_lo {
            for lx in 0..cell_w_lo {
                let mut sum = [0u32; 1];
                for dy in 0..OVERSAMPLE {
                    for dx in 0..OVERSAMPLE {
                        let hx = lx * OVERSAMPLE + dx;
                        let hy = ly * OVERSAMPLE + dy;
                        // All four channels are identical after SDF conversion;
                        // sample only the R channel and replicate on write.
                        let hi_idx = ((hy * cell_w_hi + hx) * 4) as usize;
                        sum[0] += cell_hi[hi_idx] as u32;
                    }
                }
                let ax = cell_ax_lo + lx;
                let ay = cell_ay_lo + ly;
                let lo_idx = ((ay * atlas_w + ax) * 4) as usize;
                let v = (sum[0] / block_count) as u8;
                atlas[lo_idx] = v;
                atlas[lo_idx + 1] = v;
                atlas[lo_idx + 2] = v;
                atlas[lo_idx + 3] = v;
            }
        }

        // Glyph bounding box rounded up to the low-res grid.
        let gw_hi = glyph_dims_hi[i].0;
        let gh_hi = glyph_dims_hi[i].1;

        // atlas_* stay in (supersampled) atlas texels so the UV math addresses
        // the real texture; the positional fields divide by SUPERSAMPLE too so
        // they land in logical requested-size units and on-screen layout is
        // unchanged.
        let ss_f = SUPERSAMPLE as f32;
        metrics_out.push(GlyphMetrics {
            char_code: *ch as u32,
            atlas_x: ax_hi / oversample_u16,
            atlas_y: ay_hi / oversample_u16,
            atlas_w: gw_hi / oversample_u16,
            atlas_h: gh_hi / oversample_u16,
            advance_px: metrics.advance_width / oversample_f / ss_f,
            bearing_x: metrics.xmin as f32 / oversample_f / ss_f,
            bearing_y: (metrics.ymin as f32 + gh_raw as f32) / oversample_f / ss_f,
        });
    }

    Ok(payload::serialise(
        atlas_w,
        atlas_h,
        SUPERSAMPLE,
        size_px,
        &atlas,
        &metrics_out,
    ))
}

fn round_up_to(n: u16, mult: u16) -> u16 {
    n.div_ceil(mult) * mult
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cpu::build::font::deserialise;

    #[test]
    fn builtin_face_compiles_to_a_decodable_atlas() {
        let bytes = compile(BUILTIN_FONT_BYTES, 32, "<built-in>").expect("builtin face compiles");
        let (aw, ah, supersample, size_px, rgba, metrics) =
            deserialise(&bytes).expect("payload decodes");

        assert_eq!(size_px, 32);
        assert_eq!(supersample, SUPERSAMPLE);
        assert!(aw > 0 && ah > 0);
        assert_eq!(rgba.len(), (aw * ah * 4) as usize);
        // Every printable ASCII glyph is present.
        assert_eq!(metrics.len(), (32u8..=126u8).count());
        assert!(metrics.iter().any(|m| m.char_code == 'A' as u32));
    }

    #[test]
    fn unparseable_face_reports_its_source() {
        let err = compile(b"not a font", 32, "bogus.ttf").expect_err("garbage is rejected");
        assert!(err.contains("bogus.ttf"), "error names the source: {err}");
    }

    #[test]
    fn round_up_to_snaps_to_the_next_multiple() {
        assert_eq!(round_up_to(0, 8), 0);
        assert_eq!(round_up_to(1, 8), 8);
        assert_eq!(round_up_to(8, 8), 8);
        assert_eq!(round_up_to(9, 8), 16);
    }
}

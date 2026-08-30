//! GPU-free pixel-decode math shared by the backends' frame-capture paths.
//! Turns the raw bytes read back from a GPU texture into tightly-packed opaque
//! RGBA8. The backend classifies its own format enum (MTLPixelFormat /
//! vk::Format / DXGI_FORMAT) into a `PixelLayout` and calls `decode_to_rgba8`;
//! the per-channel math here is identical across backends. PNG encoding + file
//! I/O stay in the backends (debug-only, and would pull std::fs / the png crate
//! into core).

use crate::math::{powf, powi};
use alloc::vec::Vec;

/// The decode layout of the raw read-back bytes, classified from the backend's
/// own swapchain / texture format. Carries everything the decoder needs so the
/// math stays free of any backend format enum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelLayout {
    /// 8-bit BGRA (the common SDR swapchain on Windows / macOS); channels are
    /// swizzled to RGBA and alpha is forced opaque.
    Bgra8,
    /// 8-bit RGBA; passes through with alpha forced opaque.
    Rgba8,
    /// Four IEEE 754 halfs (8 B/px). `scrgb` true applies the sRGB OETF to the
    /// linear extended-range values; false passes PQ code values through clamped.
    Rgba16F {
        /// Apply the sRGB OETF to the linear extended-range values.
        scrgb: bool,
    },
    /// Packed 2-10-10-10 unorm (one little-endian u32 per texel, R in the low 10
    /// bits); the PQ fallback swapchain. Not display-ready, but a valid image.
    A2b10g10r10,
}

/// Convert the tightly-packed read-back bytes to opaque RGBA8, decoding per the
/// classified layout. The alpha channel is forced to 255 so a saved image is
/// opaque regardless of the composited alpha.
pub fn decode_to_rgba8(raw: &[u8], layout: PixelLayout) -> Vec<u8> {
    match layout {
        PixelLayout::Bgra8 => decode_8bit(raw, true),
        PixelLayout::Rgba8 => decode_8bit(raw, false),
        PixelLayout::Rgba16F { scrgb } => decode_rgba16f(raw, scrgb),
        PixelLayout::A2b10g10r10 => decode_a2b10g10r10(raw),
    }
}

// 8-bit-per-channel formats (4 B/px). `bgra` swizzles B and R; alpha is forced
// opaque.
fn decode_8bit(raw: &[u8], bgra: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for px in raw.chunks_exact(4) {
        if bgra {
            out.extend_from_slice(&[px[2], px[1], px[0], 255]);
        } else {
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
    out
}

// `RGBA16Float` HDR read-back (8 B/px, four halfs RGBA). On the scRGB-linear
// path the stored values are linear extended-range (1.0 = SDR white), so apply
// the sRGB OETF to get a valid (non-tonemapped) image. On the PQ path the stored
// values are PQ code values already in [0, 1]; pass them through clamped.
fn decode_rgba16f(raw: &[u8], scrgb: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for px in raw.chunks_exact(8) {
        let r = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
        let g = f16_to_f32(u16::from_le_bytes([px[2], px[3]]));
        let b = f16_to_f32(u16::from_le_bytes([px[4], px[5]]));
        if scrgb {
            out.extend_from_slice(&[
                linear_to_srgb8(r),
                linear_to_srgb8(g),
                linear_to_srgb8(b),
                255,
            ]);
        } else {
            out.extend_from_slice(&[unorm_to_u8(r), unorm_to_u8(g), unorm_to_u8(b), 255]);
        }
    }
    out
}

// `A2B10G10R10_UNORM_PACK32` PQ fallback (4 B/px, one little-endian u32 per
// texel: R in bits [9:0], G [19:10], B [29:20], A [31:30]). The values are PQ
// code values, so this is not display-ready, but it is a valid image.
fn decode_a2b10g10r10(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for px in raw.chunks_exact(4) {
        let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
        let r = v & 0x3ff;
        let g = (v >> 10) & 0x3ff;
        let b = (v >> 20) & 0x3ff;
        out.extend_from_slice(&[u10_to_u8(r), u10_to_u8(g), u10_to_u8(b), 255]);
    }
    out
}

// Decode an IEEE 754 half (binary16) to f32. Handles zero, subnormals, normals,
// and inf/NaN.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    let val = match exp {
        0 => mant * powi(2.0, -24),
        0x1f => {
            if mant == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant / 1024.0) * powi(2.0, exp as i32 - 15),
    };
    sign * val
}

// sRGB OETF (linear -> display), clamped and quantised to 8-bit. NaN maps to 0.
fn linear_to_srgb8(c: f32) -> u8 {
    if c.is_nan() {
        return 0;
    }
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * powf(c, 1.0 / 2.4) - 0.055
    };
    unorm_to_u8(s)
}

// Quantise a [0, 1] value to 8-bit with rounding. NaN maps to 0.
fn unorm_to_u8(c: f32) -> u8 {
    if c.is_nan() {
        return 0;
    }
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// Scale a 10-bit unsigned value (0..=1023) to 8-bit with rounding.
fn u10_to_u8(v: u32) -> u8 {
    ((v * 255 + 511) / 1023) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn f16_round_trips_reference_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x3800), 0.5); // 0.5
        assert_eq!(f16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(f16_to_f32(0xbc00), -1.0); // -1.0
        assert!(f16_to_f32(0x7c00).is_infinite()); // +inf
        assert!(f16_to_f32(0x7e00).is_nan()); // NaN
    }

    #[test]
    fn bgra8_is_swizzled_and_made_opaque() {
        // One BGRA pixel (B=10, G=20, R=30, A=40) -> RGBA (30, 20, 10, 255).
        let raw = [10u8, 20, 30, 40];
        let out = decode_to_rgba8(&raw, PixelLayout::Bgra8);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn rgba8_passes_through_with_forced_alpha() {
        let raw = [30u8, 20, 10, 40];
        let out = decode_to_rgba8(&raw, PixelLayout::Rgba8);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn scrgb_float_applies_srgb_oetf() {
        // Linear 1.0 -> sRGB 255, 0.0 -> 0, 0.5 -> ~188 (1.055*0.5^(1/2.4)-0.055).
        let mut raw = Vec::new();
        for h in [0x3c00u16, 0x3800, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(&raw, PixelLayout::Rgba16F { scrgb: true });
        assert_eq!(out[0], 255); // r = linear 1.0
        assert!((out[1] as i32 - 188).abs() <= 1); // g = linear 0.5
        assert_eq!(out[2], 0); // b = linear 0.0
        assert_eq!(out[3], 255); // forced opaque
    }

    #[test]
    fn scrgb_float_clamps_out_of_range() {
        // Extended-range > 1.0 and negative clamp to white / black.
        let mut raw = Vec::new();
        for h in [0x4000u16, 0xbc00, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(&raw, PixelLayout::Rgba16F { scrgb: true });
        assert_eq!(out[0], 255); // r = 2.0 clamps high
        assert_eq!(out[1], 0); // g = -1.0 clamps low
    }

    #[test]
    fn pq_float_passes_code_values_through() {
        // PQ code values are already in [0, 1]; no sRGB OETF, just quantise.
        let mut raw = Vec::new();
        for h in [0x3c00u16, 0x3800, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(&raw, PixelLayout::Rgba16F { scrgb: false });
        assert_eq!(out[0], 255); // 1.0
        assert_eq!(out[1], 128); // 0.5 -> round(127.5)
        assert_eq!(out[2], 0); // 0.0
        assert_eq!(out[3], 255);
    }

    #[test]
    fn a2b10g10r10_unpacks_channels() {
        // R=1023, G=0, B=1023, A=3 packed little-endian.
        let v: u32 = 1023 | (1023 << 20) | (3 << 30);
        let out = decode_to_rgba8(&v.to_le_bytes(), PixelLayout::A2b10g10r10);
        assert_eq!(out, vec![255, 0, 255, 255]);
    }
}

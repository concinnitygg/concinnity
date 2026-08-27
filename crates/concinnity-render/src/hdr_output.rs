//! Backend-agnostic representation of the renderer's swapchain colour-output
//! mode. Built from the world's `PostProcessConfig.hdr_display` request plus
//! the active display's measured EDR capability (the backend supplies the
//! capability; this module is pure CPU). The result drives:
//!
//!   1. the swapchain pixel format + colour space chosen at window setup
//!      (BGRA8Unorm for SDR; RGBA16Float + extendedLinearDisplayP3 for HDR);
//!   2. whether `PostProcessParams.hdr_output` ships to the shader as `1.0`
//!      so the composite pass skips ACES + gamma + FXAA + ColorLut and emits
//!      linear extended-range values directly.
//!
//! Those two flags only ever enter `PostProcessParams` through
//! [`HdrOutputMode::post_process_params`], so nothing upstream of the display
//! negotiation can set (or clear) them.

use crate::render_types::{PostProcessParams, PostProcessTunables};

// Threshold above which the OS-reported max-EDR multiplier is considered an
// HDR display. macOS reports `1.0` on every panel including SDR ones; values
// above that mean the panel can drive luminance past the SDR reference white,
// so 1.0 + epsilon is the minimum useful HDR signal. Most HDR400 displays
// report 2.0+; HDR1000 displays report 8.0+.
pub(crate) const HDR_MAX_EDR_FLOOR: f32 = 1.001;

/// HDR encoding the composite shader emits on the EDR path. Drives both
/// the swapchain colour-space choice (CAMetalLayer on Metal,
/// `SetColorSpace1` on DirectX) and the shader's per-pixel encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrEncoding {
    /// Pass linear extended-range values through. Swapchain colour space is
    /// `kCGColorSpaceExtendedLinearDisplayP3`; the OS compositor handles the
    /// final encode to whatever the panel needs. `1.0` = SDR reference white;
    /// values above drive the panel's headroom.
    ExtendedLinear,
    /// PQ-encode (SMPTE ST 2084) the linear scene before write. Swapchain
    /// colour space is `kCGColorSpaceDisplayP3_PQ`; the panel decodes via
    /// the PQ EOTF. Suitable for HDR10 / HDR1000 monitors that prefer
    /// PQ-encoded values directly. SDR reference white maps to 203 nits per
    /// ITU-R BT.2408.
    Pq,
}

/// Resolved swapchain colour-output mode. Threaded into Metal + DirectX at
/// init (both honour `Hdr` end-to-end, including the PQ-encoded branch);
/// Vulkan honours the `ExtendedLinear` `Hdr` arm but ignores the PQ
/// encoding flag and falls back to SDR on a panel that reports no EDR
/// headroom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HdrOutputMode {
    /// Tone-map + gamma-encode the HDR scene into the standard BGRA8Unorm
    /// swapchain. FXAA + ColorLut run.
    Sdr,
    /// Drive an EDR-capable swapchain (`RGBA16Float`, Display P3 family
    /// colour space, `wantsExtendedDynamicRangeContent = true`). The
    /// composite shader's `hdr_output` branch skips the tonemap, gamma
    /// encode, FXAA, and LUT. `encoding` picks between scRGB-linear
    /// passthrough and PQ-encoded output. `max_edr` is the panel-reported
    /// headroom (e.g. 2.0 on an HDR400 panel, 8.0+ on HDR1000), surfaced
    /// via the StatHud `EDR` chip.
    Hdr {
        /// Reported maximum extended-range colour-component multiplier; SDR
        /// reference white is 1.0, so values above that drive HDR.
        max_edr: f32,
        /// Whether the composite shader emits PQ-encoded values or linear
        /// extended-range values. Drives both the colour-space tag and the
        /// shader branch.
        encoding: HdrEncoding,
    },
}

impl HdrOutputMode {
    /// Build the mode from the world's authored request and the platform's
    /// measured EDR multiplier. The asset toggle is the gate: even on a
    /// capable display, no HDR unless `hdr_display = true`. The reverse
    /// (`hdr_display = true` on an SDR panel) falls back to [`Self::Sdr`]
    /// and is logged once by the backend. `pq_requested` is honoured only
    /// when HDR resolves to on; off-by-default keeps the existing
    /// extended-linear path as the safer fallback.
    pub fn resolve(hdr_display_requested: bool, pq_requested: bool, max_edr: f32) -> Self {
        if hdr_display_requested && max_edr.is_finite() && max_edr >= HDR_MAX_EDR_FLOOR {
            let encoding = if pq_requested {
                HdrEncoding::Pq
            } else {
                HdrEncoding::ExtendedLinear
            };
            Self::Hdr { max_edr, encoding }
        } else {
            Self::Sdr
        }
    }

    /// Value to push into `PostProcessParams.hdr_output` so the composite
    /// shader's `> 0.5` branch lights up on the HDR path and stays inert
    /// on the SDR path.
    pub fn shader_flag(&self) -> f32 {
        match self {
            Self::Sdr => 0.0,
            Self::Hdr { .. } => 1.0,
        }
    }

    /// PQ branch value pushed into `PostProcessParams.pq_output`. The shader
    /// reads it inside its `hdr_output > 0.5` branch and switches between
    /// linear-passthrough and PQ-encode. Always `0.0` on the SDR path.
    pub fn pq_flag(&self) -> f32 {
        matches!(
            self,
            Self::Hdr {
                encoding: HdrEncoding::Pq,
                ..
            }
        ) as i32 as f32
    }

    /// True when the renderer is on the HDR path. Cheap predicate for log
    /// messages + the runtime's `hdr_display=on/off` summary line.
    pub fn is_hdr(&self) -> bool {
        matches!(self, Self::Hdr { .. })
    }

    /// Compose the GPU-facing composite uniform from the authored tunables and
    /// this negotiated mode. The backends call it once at init; afterwards a
    /// live tunable push goes through `PostProcessParams::set_tunables`, which
    /// leaves the two flags stamped here alone.
    pub fn post_process_params(&self, tunables: PostProcessTunables) -> PostProcessParams {
        PostProcessParams {
            bloom_intensity: tunables.bloom_intensity,
            bloom_threshold: tunables.bloom_threshold,
            bloom_knee: tunables.bloom_knee,
            exposure: tunables.exposure,
            vignette: tunables.vignette,
            lut_strength: tunables.lut_strength,
            hdr_output: self.shader_flag(),
            pq_output: self.pq_flag(),
            fxaa: tunables.fxaa,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_request_always_resolves_to_sdr() {
        assert_eq!(
            HdrOutputMode::resolve(false, false, 1.0),
            HdrOutputMode::Sdr
        );
        // An SDR request stays SDR even on a capable display.
        assert_eq!(
            HdrOutputMode::resolve(false, false, 8.0),
            HdrOutputMode::Sdr
        );
        // The PQ flag is ignored when HDR itself is off.
        assert_eq!(HdrOutputMode::resolve(false, true, 8.0), HdrOutputMode::Sdr);
    }

    #[test]
    fn hdr_request_on_sdr_display_falls_back_to_sdr() {
        // Apple panels report exactly 1.0 on SDR displays; clamp at floor.
        assert_eq!(HdrOutputMode::resolve(true, false, 1.0), HdrOutputMode::Sdr);
        assert_eq!(HdrOutputMode::resolve(true, false, 0.5), HdrOutputMode::Sdr);
    }

    #[test]
    fn hdr_request_on_capable_display_defaults_to_extended_linear() {
        match HdrOutputMode::resolve(true, false, 2.0) {
            HdrOutputMode::Hdr { max_edr, encoding } => {
                assert!((max_edr - 2.0).abs() < 1e-6);
                assert_eq!(encoding, HdrEncoding::ExtendedLinear);
            }
            other => panic!("expected Hdr, got {:?}", other),
        }
        assert!(HdrOutputMode::resolve(true, false, 8.0).is_hdr());
    }

    #[test]
    fn pq_request_on_capable_display_resolves_to_pq() {
        match HdrOutputMode::resolve(true, true, 8.0) {
            HdrOutputMode::Hdr { max_edr, encoding } => {
                assert!((max_edr - 8.0).abs() < 1e-6);
                assert_eq!(encoding, HdrEncoding::Pq);
            }
            other => panic!("expected Hdr/Pq, got {:?}", other),
        }
    }

    #[test]
    fn non_finite_max_edr_falls_back_to_sdr() {
        assert_eq!(
            HdrOutputMode::resolve(true, false, f32::NAN),
            HdrOutputMode::Sdr
        );
        assert_eq!(
            HdrOutputMode::resolve(true, false, f32::INFINITY),
            HdrOutputMode::Sdr
        );
    }

    #[test]
    fn shader_flag_matches_mode() {
        assert_eq!(HdrOutputMode::Sdr.shader_flag(), 0.0);
        assert_eq!(
            HdrOutputMode::Hdr {
                max_edr: 4.0,
                encoding: HdrEncoding::ExtendedLinear,
            }
            .shader_flag(),
            1.0
        );
    }

    #[test]
    fn pq_flag_is_set_only_on_the_pq_branch() {
        assert_eq!(HdrOutputMode::Sdr.pq_flag(), 0.0);
        assert_eq!(
            HdrOutputMode::Hdr {
                max_edr: 4.0,
                encoding: HdrEncoding::ExtendedLinear,
            }
            .pq_flag(),
            0.0
        );
        assert_eq!(
            HdrOutputMode::Hdr {
                max_edr: 8.0,
                encoding: HdrEncoding::Pq,
            }
            .pq_flag(),
            1.0
        );
    }

    // The one place the output flags enter the composite uniform: the tunables
    // come through untouched, the two flags come from the negotiated mode.
    #[test]
    fn composed_params_carry_the_tunables_and_the_modes_flags() {
        let tunables = PostProcessTunables {
            exposure: 4.0,
            vignette: 0.25,
            ..PostProcessTunables::DEFAULT
        };

        let sdr = HdrOutputMode::Sdr.post_process_params(tunables);
        assert_eq!(sdr.exposure, 4.0);
        assert_eq!(sdr.vignette, 0.25);
        assert_eq!(sdr.hdr_output, 0.0);
        assert_eq!(sdr.pq_output, 0.0);

        let hdr = HdrOutputMode::Hdr {
            max_edr: 8.0,
            encoding: HdrEncoding::Pq,
        }
        .post_process_params(tunables);
        assert_eq!(hdr.exposure, 4.0);
        assert_eq!(hdr.hdr_output, 1.0);
        assert_eq!(hdr.pq_output, 1.0);
    }

    // The live-push path a settings slider drives: it may move any tunable, but
    // it cannot drop the EDR path the display negotiation stamped in.
    #[test]
    fn a_tunable_push_leaves_the_negotiated_output_flags_standing() {
        let mode = HdrOutputMode::Hdr {
            max_edr: 8.0,
            encoding: HdrEncoding::Pq,
        };
        let mut params = mode.post_process_params(PostProcessTunables::DEFAULT);

        // What the engine resolves and pushes on a slider drag: no output flags
        // to speak of, so the SDR fallback is not reachable from here.
        params.set_tunables(PostProcessTunables {
            exposure: 0.5,
            bloom_intensity: 0.0,
            fxaa: 0.0,
            ..PostProcessTunables::DEFAULT
        });

        assert_eq!(params.exposure, 0.5);
        assert_eq!(params.bloom_intensity, 0.0);
        assert_eq!(params.fxaa, 0.0);
        assert_eq!(params.hdr_output, 1.0);
        assert_eq!(params.pq_output, 1.0);
    }
}

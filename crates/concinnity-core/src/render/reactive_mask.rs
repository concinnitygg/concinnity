//! The reactive mask's per-frame plan: how each pass that writes it treats it,
//! and whether the temporal passes may read it.
//!
//! The mask is a single-channel render-resolution target the particle and
//! transparent passes write beside the scene. A pixel's value is the share of
//! its color that does not follow the motion vector under it, so 1 tells TAA or
//! an upscaler to trust the current frame there over its history:
//!
//! - a particle writes its coverage, the sprite's alpha, or the light it adds
//!   (alpha times the luminance of its color) where that is larger, so a
//!   bright spark fading out still marks its pixels;
//! - glass writes its coverage times the Fresnel weight of its reflection, the
//!   part of the pane that moves with neither the pane nor what lies behind it;
//! - water writes its sun glint's luminance or half its Fresnel weight, the
//!   share its reflection and refraction take, where that is larger. The
//!   Fresnel part fades out between 4 and 16 meters from the camera: a grazing
//!   view's Fresnel weight nears 1, and its distant waves need the history
//!   most.
//!
//! Writers max-blend, so the most reactive layer over a pixel wins, and store at
//! most [`WRITE_MAX`]. The writers' pipelines always carry the target; a frame
//! with no temporal reader leaves it unstored, and a frame with no writer tells
//! the readers there is none. A reader that can cap how much history the mask
//! takes away says where ([`ReactiveReader::cap`]).

use super::render_graph::FrameGraphInputs;

/// The most any writer stores. A pixel that keeps no history aliases and
/// crawls with the jitter, and AMD recommends capping FSR's reactive input near
/// 0.9; the readers that cannot cap the mask themselves see it capped here.
/// Matches `REACTIVE_WRITE_MAX` in `shaders/reactive_mask.hlsl`.
pub const WRITE_MAX: f32 = 0.9;

/// A temporal pass that may read the reactive mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReactiveReader {
    /// The engine's TAA resolve.
    Taa,
    /// AMD FidelityFX Super Resolution.
    Fsr,
    /// Intel XeSS, through its responsive pixel mask.
    Xess,
    /// Apple MetalFX's temporal scaler.
    MetalFx,
    /// NVIDIA DLSS Super Resolution, which has no reactive input on its current
    /// models: its bias-current-color mask is limited to a retired preset.
    Dlss,
}

impl ReactiveReader {
    /// Whether it takes the mask at all.
    pub const fn reads(self) -> bool {
        !matches!(self, ReactiveReader::Dlss)
    }

    /// Whether every dispatch must name a mask, so a frame with no writer hands
    /// it a cleared one.
    pub const fn requires(self) -> bool {
        matches!(self, ReactiveReader::Xess)
    }

    /// The mask value past which it removes no more history, for a reader
    /// that takes one; the others read the mask as written. TAA keeps a fifth
    /// of its history even where the mask is full, so distant water and the
    /// aliased edge of a pane still converge; XeSS clips at the write cap.
    pub const fn cap(self) -> Option<f32> {
        match self {
            ReactiveReader::Taa => Some(0.8),
            ReactiveReader::Xess => Some(WRITE_MAX),
            ReactiveReader::Fsr | ReactiveReader::MetalFx | ReactiveReader::Dlss => None,
        }
    }
}

/// What one writer pass does with the mask attachment this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReactiveWrite {
    /// Nothing reads the mask this frame: the attachment is neither loaded nor
    /// stored, so its writes cost no memory traffic.
    Unstored,
    /// The first writer this frame: clear to 0, then store.
    Clear,
    /// A later writer: keep what an earlier writer stored, then store.
    Load,
}

impl ReactiveWrite {
    /// Whether this pass's writes reach memory.
    pub fn stores(self) -> bool {
        self != ReactiveWrite::Unstored
    }
}

/// The frame's plan for the reactive mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReactiveMaskPlan {
    /// How the particle pass treats the mask.
    pub particles: ReactiveWrite,
    /// How the transparent pass treats the mask.
    pub transparent: ReactiveWrite,
    /// Whether TAA or the upscaler reads the mask this frame.
    pub readable: bool,
}

impl ReactiveMaskPlan {
    /// The plan the frame graph built from `inputs` follows. The particle pass
    /// precedes the transparent one, so it clears whenever it runs.
    pub fn of(inputs: &FrameGraphInputs) -> Self {
        if !inputs.reactive_mask_live() {
            return Self {
                particles: ReactiveWrite::Unstored,
                transparent: ReactiveWrite::Unstored,
                readable: false,
            };
        }
        let transparent = if inputs.particles_enabled {
            ReactiveWrite::Load
        } else {
            ReactiveWrite::Clear
        };
        Self {
            particles: ReactiveWrite::Clear,
            transparent,
            readable: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(particles: bool, transparent: bool, taa: bool) -> FrameGraphInputs {
        FrameGraphInputs {
            reactive_mask_enabled: true,
            particles_enabled: particles,
            transparent_enabled: transparent,
            taa_enabled: taa,
            ..FrameGraphInputs::all_off()
        }
    }

    #[test]
    fn the_particle_pass_clears_and_the_transparent_pass_keeps_its_writes() {
        let plan = ReactiveMaskPlan::of(&inputs(true, true, true));
        assert_eq!(plan.particles, ReactiveWrite::Clear);
        assert_eq!(plan.transparent, ReactiveWrite::Load);
        assert!(plan.readable);
    }

    #[test]
    fn the_transparent_pass_clears_when_no_particle_pass_runs() {
        let plan = ReactiveMaskPlan::of(&inputs(false, true, true));
        assert_eq!(plan.transparent, ReactiveWrite::Clear);
        assert!(plan.readable);
    }

    // Without a reader every writer leaves the mask unstored.
    #[test]
    fn a_frame_with_no_temporal_reader_stores_nothing() {
        let plan = ReactiveMaskPlan::of(&inputs(true, true, false));
        assert!(!plan.particles.stores() && !plan.transparent.stores());
        assert!(!plan.readable);
    }

    #[test]
    fn every_reader_caps_the_mask_within_what_is_written() {
        for reader in [
            ReactiveReader::Taa,
            ReactiveReader::Fsr,
            ReactiveReader::Xess,
            ReactiveReader::MetalFx,
            ReactiveReader::Dlss,
        ] {
            if let Some(cap) = reader.cap() {
                assert!(reader.reads(), "{reader:?}");
                assert!(cap > 0.0 && cap <= WRITE_MAX, "{reader:?}");
            }
            assert!(!reader.requires() || reader.reads(), "{reader:?}");
        }
        assert!(ReactiveReader::Taa.cap().is_some_and(|cap| cap < WRITE_MAX));
    }

    #[test]
    fn the_write_cap_matches_the_shader() {
        assert!(
            crate::render::shaders::REACTIVE_MASK.contains(&alloc::format!(
                "static const float REACTIVE_WRITE_MAX = {WRITE_MAX:?};"
            ))
        );
    }

    #[test]
    fn a_hidden_world_writes_no_mask() {
        let hidden = FrameGraphInputs {
            world_hidden: true,
            ..inputs(true, true, true)
        };
        assert!(!ReactiveMaskPlan::of(&hidden).readable);
    }
}

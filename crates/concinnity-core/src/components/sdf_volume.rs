//! Runtime behavior for the SdfVolume asset. The authored schema (the
//! SdfVolume struct, its Default, `cone_ratio`, and `SDF_PARAMS_LEN`) lives in
//! concinnity-asset; this file keeps the `Component` impl, the blob-residency
//! helper the engine init uses, and the runtime step-count clamp bounds. The
//! JSON-args source selection, validation, and the bake-time clamp live in
//! concinnity-world (`source_args`, `check::sdf_volume`, `validate::sdf_volume`).
//! The schema type + `SDF_PARAMS_LEN` are re-exported so
//! `crate::components::sdf_volume::*` paths (the render backends' uniform structs)
//! keep resolving.

pub use concinnity_asset::{SDF_PARAMS_LEN, SdfVolume};

use crate::ecs::asset_id::AssetId;
use crate::ecs::{Component, PayloadLocator};

/// Hard cap on the per-volume cone-march step count. Matches the
/// runtime kernel's loop bound; values above this are clamped.
pub const SDF_MAX_STEPS_CEILING: u32 = 256;

/// Lower bound on the per-volume cone-march step count. Below this the
/// march doesn't have enough budget to converge on anything interesting.
pub const SDF_MAX_STEPS_FLOOR: u32 = 8;

impl Component for SdfVolume {
    const NAME: &'static str = "SdfVolume";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(crate::blob::decode_exact(bytes)?)
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
}

/// Blob indices that hold an `SdfVolume` fragment-shader payload.
///
/// The graphics-system init drains `SdfVolume`s and reads their payload
/// bytes via the locator. The release sweep earlier in the same init
/// frees every blob whose contents have already been consumed, but
/// because the SDF drain runs *after* that sweep, any blob holding only
/// an SDF payload would be freed before being read. (When the world
/// has other small assets, the SDF shader bytes typically share a blob
/// with a kept asset and survive by accident; a world whose SDF shader
/// ends up alone in its blob exposes the bug as "SdfVolume payload
/// FileIo, skipping" with no surface drawn.) This helper lets the
/// release sweep keep SDF blobs resident, matching the
/// `audio_clip_blob_indices` pattern.
pub fn sdf_volume_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> alloc::collections::BTreeSet<u32> {
    ctx.query::<SdfVolume>()
        .filter_map(|v| v.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let v = SdfVolume::default();
        assert_eq!(v.centre, [0.0, 0.0, 0.0]);
        assert_eq!(v.extent, [1.0, 1.0, 1.0]);
        assert_eq!(v.max_gradient, 1.0);
        assert_eq!(v.max_steps, 64);
        assert_eq!(v.max_distance, 30.0);
        assert!(v.receive_shadows);
        assert!(!v.cast_shadows);
        assert!(v.visible);
        assert_eq!(v.params.len(), SDF_PARAMS_LEN);
        assert_eq!(v.cone_ratio(), 1.0);
    }

    #[test]
    fn cone_ratio_inverts_gradient() {
        let v = SdfVolume {
            max_gradient: 2.0,
            ..Default::default()
        };
        assert!((v.cone_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn volumetric_default_is_off() {
        let v = SdfVolume::default();
        assert!(!v.volumetric);
    }
}

//! What changed between two resolved snapshots, grouped by when each change
//! takes effect. A resolved value is named by the authored field it resolves
//! (`AmbientOcclusion.enabled`), and takes that field's apply class; the few
//! the world does not author are named under `UserGraphicsSettings`.

use alloc::vec::Vec;

use crate::components::{
    AntiAliasing, Bloom, ColorLut, Culling, Display, Exposure, FramePacing, IndirectLighting,
    Shadows, StreamingConfig, TextureFiltering, Vignette,
};
use crate::ecs::asset_fields::{ApplyClass, AssetFields, ParamChange};

use super::ResolvedSettings;

const USER: &str = "UserGraphicsSettings";

/// The parameters that differ between two resolved snapshots, each once, in
/// a list per apply class.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SettingsDiff {
    by_class: [Vec<ParamChange>; 4],
}

impl SettingsDiff {
    /// The changes from `before` to `after`.
    pub fn between(before: &ResolvedSettings, after: &ResolvedSettings) -> Self {
        let mut changes = Vec::new();
        let (b, a) = (before, after);
        let (eb, ea) = (&b.quality.effects, &a.quality.effects);
        ea.ambient_occlusion
            .changed_params(&eb.ambient_occlusion, &mut changes);
        ea.reflections.changed_params(&eb.reflections, &mut changes);
        ea.indirect_lighting
            .changed_params(&eb.indirect_lighting, &mut changes);
        ea.anti_aliasing
            .changed_params(&eb.anti_aliasing, &mut changes);
        ea.exposure.changed_params(&eb.exposure, &mut changes);

        let mut changed = |asset: Asset, key: &'static str, changed: bool| {
            if changed {
                asset.push(&mut changes, key);
            }
        };
        changed(
            Asset::AntiAliasing,
            "upscale_quality",
            a.quality.render_scale != b.quality.render_scale,
        );
        let (tb, ta) = (&b.post_process, &a.post_process);
        changed(Asset::Exposure, "ev", ta.exposure != tb.exposure);
        changed(
            Asset::Bloom,
            "intensity",
            ta.bloom_intensity != tb.bloom_intensity,
        );
        changed(
            Asset::Bloom,
            "threshold",
            ta.bloom_threshold != tb.bloom_threshold,
        );
        changed(Asset::Bloom, "knee", ta.bloom_knee != tb.bloom_knee);
        changed(Asset::Vignette, "strength", ta.vignette != tb.vignette);
        changed(
            Asset::ColorLut,
            "strength",
            ta.lut_strength != tb.lut_strength,
        );
        changed(Asset::AntiAliasing, "mode", ta.fxaa != tb.fxaa);
        changed(
            Asset::IndirectLighting,
            "ambient_intensity",
            a.ambient_intensity != b.ambient_intensity,
        );
        changed(Asset::Display, "hdr", a.hdr_display != b.hdr_display);
        changed(Asset::Display, "pq", a.hdr_pq != b.hdr_pq);
        changed(
            Asset::AntiAliasing,
            "upscaling",
            a.temporal_upscaling != b.temporal_upscaling,
        );
        changed(
            Asset::AntiAliasing,
            "upscaler",
            a.upscale_backend != b.upscale_backend,
        );
        changed(
            Asset::Culling,
            "two_pass_occlusion",
            a.occlusion_two_pass != b.occlusion_two_pass,
        );

        let (qb, qa) = (&b.quality, &a.quality);
        changed(
            Asset::Shadows,
            "map_size",
            qa.shadow_map_size != qb.shadow_map_size,
        );
        changed(
            Asset::Shadows,
            "update",
            qa.shadow_cadence.update != qb.shadow_cadence.update,
        );
        changed(
            Asset::Shadows,
            "distance",
            qa.shadow_cadence.distance != qb.shadow_cadence.distance,
        );
        changed(
            Asset::Shadows,
            "cascades",
            qa.shadow_cadence.cascades != qb.shadow_cadence.cascades,
        );
        changed(
            Asset::TextureFiltering,
            "anisotropy",
            qa.anisotropy != qb.anisotropy,
        );
        changed(Asset::FramePacing, "vsync", a.vsync != b.vsync);
        changed(Asset::FramePacing, "fps_cap", a.fps_cap != b.fps_cap);
        changed(
            Asset::FramePacing,
            "frames_in_flight",
            a.frames_in_flight != b.frames_in_flight,
        );

        changed(
            Asset::StreamingConfig,
            "texture_cap",
            a.texture_cap != b.texture_cap,
        );
        changed(
            Asset::StreamingConfig,
            "texture_budget",
            a.texture_budget != b.texture_budget,
        );

        // Values only the user sets: the DLSS preset is chosen when the
        // upscaler is built, and the stats overlay reads its toggles each frame.
        let mut user = |key: &'static str, apply: ApplyClass, changed: bool| {
            if changed {
                changes.push(ParamChange {
                    asset: USER,
                    key,
                    apply,
                });
            }
        };
        user(
            "dlss_preset",
            ApplyClass::Restart,
            a.dlss_preset != b.dlss_preset,
        );
        user("perf_stats", ApplyClass::Live, a.perf_stats != b.perf_stats);
        user("show_fps", ApplyClass::Live, a.show_fps != b.show_fps);
        user("show_vram", ApplyClass::Live, a.show_vram != b.show_vram);

        let mut diff = Self::default();
        for change in changes {
            let class = &mut diff.by_class[class_slot(change.apply)];
            if !class.contains(&change) {
                class.push(change);
            }
        }
        diff
    }

    /// The changes that take effect the way `class` describes.
    pub fn class(&self, class: ApplyClass) -> &[ParamChange] {
        &self.by_class[class_slot(class)]
    }

    /// Every change, cheapest class first.
    pub fn changes(&self) -> impl Iterator<Item = &ParamChange> {
        self.by_class.iter().flatten()
    }

    /// Whether nothing changed.
    pub fn is_empty(&self) -> bool {
        self.by_class.iter().all(Vec::is_empty)
    }

    /// The most expensive class any change needs, or `None` when nothing
    /// changed.
    pub fn costliest(&self) -> Option<ApplyClass> {
        ApplyClass::ALL
            .into_iter()
            .rev()
            .find(|&class| !self.class(class).is_empty())
    }
}

fn class_slot(class: ApplyClass) -> usize {
    match class {
        ApplyClass::Live => 0,
        ApplyClass::Rebuild => 1,
        ApplyClass::Variant => 2,
        ApplyClass::Restart => 3,
    }
}

// The assets whose resolved values the diff compares outside the
// quality-governed effects.
#[derive(Clone, Copy)]
enum Asset {
    AntiAliasing,
    Exposure,
    Bloom,
    Vignette,
    ColorLut,
    IndirectLighting,
    Display,
    Culling,
    Shadows,
    TextureFiltering,
    FramePacing,
    StreamingConfig,
}

impl Asset {
    // Record a change to this asset's field `key` under the apply class it
    // declares.
    fn push(self, changes: &mut Vec<ParamChange>, key: &'static str) {
        match self {
            Self::AntiAliasing => push::<AntiAliasing>(changes, "AntiAliasing", key),
            Self::Exposure => push::<Exposure>(changes, "Exposure", key),
            Self::Bloom => push::<Bloom>(changes, "Bloom", key),
            Self::Vignette => push::<Vignette>(changes, "Vignette", key),
            Self::ColorLut => push::<ColorLut>(changes, "ColorLut", key),
            Self::IndirectLighting => push::<IndirectLighting>(changes, "IndirectLighting", key),
            Self::Display => push::<Display>(changes, "Display", key),
            Self::Culling => push::<Culling>(changes, "Culling", key),
            Self::Shadows => push::<Shadows>(changes, "Shadows", key),
            Self::TextureFiltering => push::<TextureFiltering>(changes, "TextureFiltering", key),
            Self::FramePacing => push::<FramePacing>(changes, "FramePacing", key),
            Self::StreamingConfig => push::<StreamingConfig>(changes, "StreamingConfig", key),
        }
    }
}

// Record a change to `A`'s field `key` under the apply class it declares. A
// field declaring none is recorded as needing a restart, the class that never
// under-promises.
fn push<A: AssetFields>(changes: &mut Vec<ParamChange>, asset: &'static str, key: &'static str) {
    let apply = A::param(key)
        .and_then(|meta| meta.apply)
        .unwrap_or(ApplyClass::Restart);
    changes.push(ParamChange { asset, key, apply });
}

//! What changed between two resolved snapshots, grouped by when each change
//! takes effect. A resolved value is named by the authored field it resolves
//! (`PostProcessConfig.ssao`), and takes that field's apply class; the few the
//! world does not author are named under `UserGraphicsSettings`.

use alloc::vec::Vec;

use crate::components::{GraphicsConfig, PostProcessConfig, StreamingConfig};
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
        a.quality
            .post_config
            .changed_params(&b.quality.post_config, &mut changes);

        let mut post = |key: &'static str, changed: bool| {
            if changed {
                push::<PostProcessConfig>(&mut changes, "PostProcessConfig", key);
            }
        };
        post(
            "upscale_quality",
            a.quality.render_scale != b.quality.render_scale,
        );
        let (tb, ta) = (&b.post_process, &a.post_process);
        post("exposure_ev", ta.exposure != tb.exposure);
        post("bloom_intensity", ta.bloom_intensity != tb.bloom_intensity);
        post("bloom_threshold", ta.bloom_threshold != tb.bloom_threshold);
        post("bloom_knee", ta.bloom_knee != tb.bloom_knee);
        post("vignette_strength", ta.vignette != tb.vignette);
        post("lut_strength", ta.lut_strength != tb.lut_strength);
        post("aa_mode", ta.fxaa != tb.fxaa);
        post(
            "ambient_intensity",
            a.ambient_intensity != b.ambient_intensity,
        );
        post("hdr_display", a.hdr_display != b.hdr_display);
        post("hdr_pq", a.hdr_pq != b.hdr_pq);
        post(
            "temporal_upscaling",
            a.temporal_upscaling != b.temporal_upscaling,
        );
        post("upscale_backend", a.upscale_backend != b.upscale_backend);
        post(
            "occlusion_two_pass",
            a.occlusion_two_pass != b.occlusion_two_pass,
        );

        let mut graphics = |key: &'static str, changed: bool| {
            if changed {
                push::<GraphicsConfig>(&mut changes, "GraphicsConfig", key);
            }
        };
        let (qb, qa) = (&b.quality, &a.quality);
        graphics("shadow_map_size", qa.shadow_map_size != qb.shadow_map_size);
        graphics(
            "shadow_update",
            qa.shadow_cadence.update != qb.shadow_cadence.update,
        );
        graphics(
            "shadow_distance",
            qa.shadow_cadence.distance != qb.shadow_cadence.distance,
        );
        graphics(
            "shadow_cascades",
            qa.shadow_cadence.cascades != qb.shadow_cadence.cascades,
        );
        graphics("anisotropy", qa.anisotropy != qb.anisotropy);
        graphics("vsync", a.vsync != b.vsync);
        graphics("fps_cap", a.fps_cap != b.fps_cap);
        graphics("frames_in_flight", a.frames_in_flight != b.frames_in_flight);

        if a.texture_cap != b.texture_cap {
            push::<StreamingConfig>(&mut changes, "StreamingConfig", "texture_cap");
        }
        if a.texture_budget != b.texture_budget {
            push::<StreamingConfig>(&mut changes, "StreamingConfig", "texture_budget");
        }

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

// Record a change to `A`'s field `key` under the apply class it declares. A
// field declaring none is recorded as needing a restart, the class that never
// under-promises.
fn push<A: AssetFields>(changes: &mut Vec<ParamChange>, asset: &'static str, key: &'static str) {
    let apply = A::param(key)
        .and_then(|meta| meta.apply)
        .unwrap_or(ApplyClass::Restart);
    changes.push(ParamChange { asset, key, apply });
}

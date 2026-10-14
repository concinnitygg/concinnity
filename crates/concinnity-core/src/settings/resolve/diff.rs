//! What changed between two resolved snapshots, grouped by when each change
//! takes effect. A resolved value is named by the section field it resolves
//! (`AmbientOcclusion.enabled`), and takes that field's apply class; the few
//! the world does not author are named under `UserGraphicsSettings`.

use alloc::vec::Vec;

use crate::components::{ColorLut, StreamingConfig};
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
        let (b, a) = (&before.graphics, &after.graphics);
        a.ambient_occlusion
            .changed_params(&b.ambient_occlusion, &mut changes);
        a.reflections.changed_params(&b.reflections, &mut changes);
        a.indirect_lighting
            .changed_params(&b.indirect_lighting, &mut changes);
        a.anti_aliasing
            .changed_params(&b.anti_aliasing, &mut changes);
        a.exposure.changed_params(&b.exposure, &mut changes);
        a.bloom.changed_params(&b.bloom, &mut changes);
        a.vignette.changed_params(&b.vignette, &mut changes);
        if a.lut_strength != b.lut_strength {
            push::<ColorLut>(&mut changes, "ColorLut", "strength");
        }
        a.display.changed_params(&b.display, &mut changes);
        a.culling.changed_params(&b.culling, &mut changes);
        a.shadows.changed_params(&b.shadows, &mut changes);
        a.texture_filtering
            .changed_params(&b.texture_filtering, &mut changes);
        a.frame_pacing.changed_params(&b.frame_pacing, &mut changes);
        a.background.changed_params(&b.background, &mut changes);
        let streaming = |key, changed: bool, changes: &mut Vec<ParamChange>| {
            if changed {
                push::<StreamingConfig>(changes, "StreamingConfig", key);
            }
        };
        streaming(
            "texture_cap",
            after.texture_cap != before.texture_cap,
            &mut changes,
        );
        streaming(
            "texture_budget",
            after.texture_budget != before.texture_budget,
            &mut changes,
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
        user(
            "perf_stats",
            ApplyClass::Live,
            after.perf_stats != before.perf_stats,
        );
        user(
            "show_fps",
            ApplyClass::Live,
            after.show_fps != before.show_fps,
        );
        user(
            "show_vram",
            ApplyClass::Live,
            after.show_vram != before.show_vram,
        );

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

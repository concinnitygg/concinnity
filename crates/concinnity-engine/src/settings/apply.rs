//! What a settings change asks of a running renderer: [`plan_apply`] turns a
//! [`SettingsDiff`] into the [`ApplyPlan`] the renderer carries out.

use crate::ecs::asset_fields::{ApplyClass, ParamChange};
use crate::render::config::ResolvedGraphics;
use crate::render::config::apply::{ApplyPlan, LiveChanges, RebuildFeature};
use crate::settings::SettingsDiff;

/// The plan `diff` (from `before` to `after`) asks of a running renderer.
pub fn plan_apply(
    diff: &SettingsDiff,
    before: &ResolvedGraphics,
    after: &ResolvedGraphics,
) -> ApplyPlan {
    let mut plan = ApplyPlan::default();
    for change in diff.changes() {
        match change.apply {
            ApplyClass::Live => live_change(&mut plan.live, change),
            ApplyClass::Rebuild => rebuild_change(&mut plan, change),
            ApplyClass::Variant | ApplyClass::Restart => plan.restart.push(*change),
        }
    }
    if !plan.rebuild.is_empty() && needs_gbuffer(before) != needs_gbuffer(after) {
        plan.rebuild.insert(RebuildFeature::GBuffer);
    }
    plan
}

// The live group a `live` change rides.
fn live_change(live: &mut LiveChanges, change: &ParamChange) {
    match (change.asset, change.key) {
        ("Bloom" | "Vignette" | "ColorLut", _) | ("Exposure", "ev") => {
            live.post_process = true;
        }
        ("IndirectLighting", "ambient_intensity") => live.ambient = true,
        ("Shadows", _) | ("QualityCeiling", "far_shadow_refresh_frames") => {
            live.shadow_cadence = true;
        }
        ("FramePacing", "vsync") => live.vsync = true,
        ("AmbientOcclusion" | "Reflections" | "IndirectLighting" | "Exposure", _) => {
            live.feature_params = true;
        }
        // Read by the engine each frame (frame cap, stats overlay), never by
        // the renderer.
        _ => {}
    }
}

// The features a `rebuild` change reallocates, and the live values riding it.
fn rebuild_change(plan: &mut ApplyPlan, change: &ParamChange) {
    use RebuildFeature::{AutoExposure, ReflectionComposite, RtReflections, Ssao, Ssgi, Ssr, Taa};
    let rebuild = &mut plan.rebuild;
    match (change.asset, change.key) {
        ("AmbientOcclusion", "enabled") => rebuild.insert(Ssao),
        ("Reflections", "screen_space") => {
            rebuild.insert(Ssr);
            rebuild.insert(ReflectionComposite);
        }
        ("Reflections", "ray_traced") => {
            rebuild.insert(RtReflections);
            rebuild.insert(ReflectionComposite);
        }
        ("Reflections", "ray_traced_resolution" | "ray_traced_shadows")
        | ("QualityCeiling", "water_trace") => {
            rebuild.insert(RtReflections);
        }
        ("Reflections", "blur_resolution") => rebuild.insert(ReflectionComposite),
        ("AntiAliasing", "mode") => {
            rebuild.insert(Taa);
            // The composite's FXAA flag follows the mode.
            plan.live.post_process = true;
        }
        ("Exposure", "auto") => {
            rebuild.insert(AutoExposure);
            // With auto-exposure off the static EV drives exposure again.
            plan.live.post_process = true;
        }
        ("IndirectLighting", _) => rebuild.insert(Ssgi),
        // A `rebuild` field no feature claims would never apply; the plan test
        // names every one, so reaching here is a field added without a row.
        _ => plan.restart.push(*change),
    }
}

// Whether any screen-space consumer of the G-buffer pre-pass is on.
fn needs_gbuffer(g: &ResolvedGraphics) -> bool {
    g.ssao().is_some()
        || g.ssr().is_some()
        || g.ssgi().is_some()
        || g.rt_reflections().is_some()
        || g.taa()
        || g.anti_aliasing.upscaling
}

#[cfg(test)]
mod tests;

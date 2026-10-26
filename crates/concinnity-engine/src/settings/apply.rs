//! What a settings change asks of a running renderer, grouped by how it takes
//! effect: values the frame reads each draw, features whose resources
//! reallocate, and values that wait for a relaunch. [`plan_apply`] turns a
//! [`SettingsDiff`] into that plan; [`apply_plan`] carries it out on any
//! [`SettingsTarget`], in the one order every backend shares.

use alloc::vec::Vec;

use crate::ecs::asset_fields::{ApplyClass, ParamChange};
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend::error::RenderResult;
use crate::render::backend::init::ShadowCadence;
use crate::settings::{ResolvedGraphics, SettingsDiff};

/// The live values a change moved, each pushed to the frame's uniforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per independently applied settings change"
)]
pub struct LiveChanges {
    /// The composite tunables (bloom, exposure, vignette, LUT, FXAA).
    pub post_process: bool,
    /// The ambient (IBL) scale.
    pub ambient: bool,
    /// The cascade shadow schedule.
    pub shadow_cadence: bool,
    /// The per-feature sub-quality scalars (SSAO, SSR, SSGI, auto-exposure).
    pub feature_params: bool,
    /// Display sync.
    pub vsync: bool,
}

impl LiveChanges {
    /// Whether nothing moved.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A feature whose resources a settings change reallocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildFeature {
    /// The unified G-buffer pre-pass targets.
    GBuffer,
    /// The TAA resolve.
    Taa,
    /// Screen-space ambient occlusion.
    Ssao,
    /// The screen-space reflection resolve.
    Ssr,
    /// Screen-space global illumination.
    Ssgi,
    /// The ray-traced reflection pass.
    RtReflections,
    /// The roughness-aware reflection composite.
    ReflectionComposite,
    /// The auto-exposure histogram.
    AutoExposure,
}

impl RebuildFeature {
    /// Every feature.
    pub const ALL: [Self; 8] = [
        Self::GBuffer,
        Self::Taa,
        Self::Ssao,
        Self::Ssr,
        Self::Ssgi,
        Self::RtReflections,
        Self::ReflectionComposite,
        Self::AutoExposure,
    ];

    fn bit(self) -> u16 {
        1 << self as u16
    }
}

/// A set of [`RebuildFeature`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RebuildSet(u16);

impl RebuildSet {
    /// Add `feature`.
    pub fn insert(&mut self, feature: RebuildFeature) {
        self.0 |= feature.bit();
    }

    /// Whether `feature` is in the set.
    pub fn contains(&self, feature: RebuildFeature) -> bool {
        self.0 & feature.bit() != 0
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// The features in the set, in [`RebuildFeature::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = RebuildFeature> + '_ {
        RebuildFeature::ALL
            .into_iter()
            .filter(|f| self.contains(*f))
    }
}

/// What one settings change asks of a running renderer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplyPlan {
    /// The live values to push.
    pub live: LiveChanges,
    /// The features to reallocate.
    pub rebuild: RebuildSet,
    /// The changes that wait for a relaunch.
    pub restart: Vec<ParamChange>,
}

impl ApplyPlan {
    /// A plan that only pushes `live`.
    pub fn live(live: LiveChanges) -> Self {
        Self {
            live,
            ..Self::default()
        }
    }

    /// Whether the running renderer has anything to do.
    pub fn touches_renderer(&self) -> bool {
        !self.live.is_empty() || !self.rebuild.is_empty()
    }
}

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
        ("Shadows", _) => live.shadow_cadence = true,
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

/// Run `rebuild` once for each feature in `features`, in [`RebuildFeature::ALL`]
/// order: the G-buffer before the passes that read it, and the ray-traced pass
/// before the composite that follows what it built. A feature that fails keeps
/// what it had and does not stop the rest; the first error comes back once every
/// feature has run.
///
/// # Errors
///
/// Returns the first error `rebuild` returned.
pub fn rebuild_each(
    features: RebuildSet,
    mut rebuild: impl FnMut(RebuildFeature) -> RenderResult<()>,
) -> RenderResult<()> {
    let mut first = Ok(());
    for feature in features.iter() {
        if let Err(e) = rebuild(feature) {
            if first.is_ok() {
                first = Err(e);
            } else {
                tracing::error!("rebuilding {feature:?}: {e}");
            }
        }
    }
    first
}

/// The pieces of a renderer a settings plan reaches. Each backend implements
/// it once; [`apply_plan`] decides what runs and in which order.
pub trait SettingsTarget {
    /// Reallocate `features` to match `graphics`, leaving every other feature's
    /// resources untouched.
    ///
    /// # Errors
    ///
    /// Returns an error if a feature's GPU resources cannot be reallocated.
    fn rebuild_features(
        &mut self,
        graphics: &ResolvedGraphics,
        features: RebuildSet,
    ) -> RenderResult<()>;
    /// Push the per-feature sub-quality scalars of the live features.
    fn push_feature_params(&mut self, graphics: &ResolvedGraphics);
    /// Push the composite tunables.
    fn push_post_process(&mut self, tunables: PostProcessTunables);
    /// Push the ambient (IBL) scale.
    fn push_ambient(&mut self, value: f32);
    /// Push the cascade shadow schedule.
    fn push_shadow_cadence(&mut self, cadence: ShadowCadence);
    /// Turn display sync on or off.
    fn push_vsync(&mut self, on: bool);
}

/// Carry `plan` out on `target`: reallocate first, so the live pushes that
/// follow land on the rebuilt features (a static exposure once auto-exposure
/// is off, the sub-quality scalars of a feature just turned on).
///
/// # Errors
///
/// Returns an error if reallocating the features fails; the live pushes still
/// apply.
pub fn apply_plan<T: SettingsTarget + ?Sized>(
    target: &mut T,
    graphics: &ResolvedGraphics,
    plan: &ApplyPlan,
) -> RenderResult<()> {
    let rebuilt = if plan.rebuild.is_empty() {
        Ok(())
    } else {
        target.rebuild_features(graphics, plan.rebuild)
    };
    let live = &plan.live;
    if live.feature_params || !plan.rebuild.is_empty() {
        target.push_feature_params(graphics);
    }
    if live.post_process {
        target.push_post_process(graphics.post_process());
    }
    if live.ambient {
        target.push_ambient(graphics.ambient_intensity());
    }
    if live.shadow_cadence {
        target.push_shadow_cadence(graphics.shadow_cadence());
    }
    if live.vsync {
        target.push_vsync(graphics.frame_pacing.vsync);
    }
    rebuilt
}

#[cfg(test)]
mod tests;

//! How a graphics configuration change reaches a running renderer, grouped by
//! how it takes effect: values the frame reads each draw, features whose
//! resources reallocate, and values that wait for a relaunch. [`apply_plan`]
//! carries an [`ApplyPlan`] out on any [`SettingsTarget`], in the one order
//! every backend shares.

use alloc::vec::Vec;

use crate::ecs::asset_fields::ParamChange;
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::backend::init::{ShadowCadence, ShadowMapSizes};
use crate::render::config::ResolvedGraphics;

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

/// The pieces of a renderer a settings plan reaches: the device seam every
/// backend implements once, and [`apply_plan`] decides what runs and in which
/// order.
///
/// Defaulted for a backend without live settings: the pushes are feeds and do
/// nothing, so it keeps its init-time values, and a reallocation is a request
/// and reports [`RenderError::Unsupported`], so the caller learns it did not
/// happen.
pub trait SettingsTarget {
    /// Reallocate `features` to match `graphics`, leaving every other feature's
    /// resources untouched.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend with no live rebuild
    /// path, or an error if a feature's GPU resources cannot be reallocated.
    fn rebuild_features(
        &mut self,
        graphics: &ResolvedGraphics,
        features: RebuildSet,
    ) -> RenderResult<()> {
        let _ = (graphics, features);
        Err(RenderError::Unsupported {
            op: "rebuild_features",
        })
    }
    /// Push the per-feature sub-quality scalars of the live features.
    fn push_feature_params(&mut self, graphics: &ResolvedGraphics) {
        let _ = graphics;
    }
    /// Push the composite tunables.
    fn push_post_process(&mut self, tunables: PostProcessTunables) {
        let _ = tunables;
    }
    /// Push the ambient (IBL) scale.
    fn push_ambient(&mut self, value: f32) {
        let _ = value;
    }
    /// Push the cascade shadow schedule.
    fn push_shadow_cadence(&mut self, cadence: ShadowCadence) {
        let _ = cadence;
    }
    /// Turn display sync on or off.
    fn push_vsync(&mut self, on: bool) {
        let _ = on;
    }
    /// The sun's shadow maps this backend built: what the settings take as
    /// built when they key a restart or the planet's horizon reach. None by
    /// default.
    fn built_shadow_maps(&self) -> ShadowMapSizes {
        ShadowMapSizes::default()
    }
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

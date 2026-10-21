//! Which toggle-controlled features a backend builds for a set of resolved
//! graphics settings, and what a change does to each built one. Init and the
//! live quality toggle both resolve the same plan, so a toggle produces the
//! features a launch with the same settings would.

use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::settings::ResolvedGraphics;

/// What the backend reports about itself when resolving a [`FeaturePlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendFacts {
    /// The device can build the scene acceleration structure and trace it.
    pub rt_capable: bool,
    /// A temporal upscaler built and takes the frame's velocity. The one the
    /// settings request may have failed to build, and then TAA runs instead.
    pub upscaling: bool,
}

/// The toggle-controlled features the settings want built.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent render feature toggles"
)]
pub struct FeaturePlan {
    /// Ray-traced reflections, when the settings ask and the device can.
    pub rt: Option<RtReflectionSettings>,
    /// The settings asked for ray tracing the device cannot run, a capability
    /// probe and backend disagreement, logged as an error.
    pub rt_refused: bool,
    /// The jittered velocity pre-pass and its history: TAA, or an upscaler
    /// that consumes the velocity.
    pub temporal: bool,
    /// The TAA resolve itself, which an upscaler's own accumulation replaces.
    pub taa_resolve: bool,
    /// Screen-space reflections are authored.
    pub ssr: bool,
    /// The SSR resolve plumbing, which SSGI and ray tracing reuse.
    pub ssr_resources: bool,
    /// Screen-space global illumination.
    pub ssgi: bool,
    /// Screen-space ambient occlusion.
    pub ssao: bool,
    /// Automatic exposure.
    pub auto_exposure: bool,
    /// The unified G-buffer pre-pass every screen-space consumer reads.
    pub gbuffer: bool,
}

impl FeaturePlan {
    /// The plan for `graphics` on a backend described by `facts`.
    pub fn resolve(graphics: &ResolvedGraphics, facts: BackendFacts) -> Self {
        let requested_rt = graphics.rt_reflections();
        let rt = requested_rt.filter(|_| facts.rt_capable);
        let temporal = graphics.taa() || facts.upscaling;
        let ssr = graphics.ssr().is_some();
        let ssgi = graphics.ssgi().is_some();
        let ssao = graphics.ssao().is_some();
        let ssr_resources = ssr || ssgi || rt.is_some();
        let rt_refused = requested_rt.is_some() && !facts.rt_capable;
        if rt_refused {
            tracing::error!(
                "the resolved settings asked for ray-traced reflections the device cannot run; \
                 the capability probe and the backend disagree"
            );
        }
        Self {
            rt,
            rt_refused,
            temporal,
            taa_resolve: graphics.taa() && !facts.upscaling,
            ssr,
            ssr_resources,
            ssgi,
            ssao,
            auto_exposure: graphics.auto_exposure().is_some(),
            gbuffer: ssr_resources || ssao || temporal,
        }
    }
}

/// What a change does to one feature, from whether the plan wants it and
/// whether it is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureChange {
    /// Wanted and missing: build it.
    Build,
    /// Built and no longer wanted: tear it down.
    Drop,
    /// Wanted and built: keep it, taking any new tunables.
    Update,
    /// Neither wanted nor built.
    Absent,
}

impl FeatureChange {
    /// The change for a feature the plan `wanted` that is `built` or not.
    pub fn of(wanted: bool, built: bool) -> Self {
        match (wanted, built) {
            (true, false) => Self::Build,
            (false, true) => Self::Drop,
            (true, true) => Self::Update,
            (false, false) => Self::Absent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AaMode;

    #[derive(Clone, Copy, PartialEq)]
    enum Feature {
        Taa,
        Upscaling,
        Ssr,
        Rt,
    }

    fn graphics(on: &[Feature]) -> ResolvedGraphics {
        let mut g = ResolvedGraphics::default();
        g.anti_aliasing.mode = if on.contains(&Feature::Taa) {
            AaMode::Taa
        } else {
            AaMode::Off
        };
        g.anti_aliasing.upscaling = on.contains(&Feature::Upscaling);
        g.reflections.screen_space = on.contains(&Feature::Ssr);
        g.reflections.ray_traced = on.contains(&Feature::Rt);
        g
    }

    const CAPABLE: BackendFacts = BackendFacts {
        rt_capable: true,
        upscaling: false,
    };

    #[test]
    fn rt_needs_a_capable_device_and_reports_a_refusal() {
        let g = graphics(&[Feature::Rt]);
        let on = FeaturePlan::resolve(&g, CAPABLE);
        assert!(on.rt.is_some() && !on.rt_refused);
        assert!(on.ssr_resources && on.gbuffer, "rt reuses the ssr plumbing");
        let off = FeaturePlan::resolve(
            &g,
            BackendFacts {
                rt_capable: false,
                ..CAPABLE
            },
        );
        assert!(off.rt.is_none() && off.rt_refused);
        assert_eq!(off.ssr_resources, off.ssr || off.ssgi);
    }

    #[test]
    fn an_upscaler_keeps_the_velocity_and_drops_the_taa_resolve() {
        let g = graphics(&[Feature::Taa]);
        let plain = FeaturePlan::resolve(&g, CAPABLE);
        assert!(plain.temporal && plain.taa_resolve && plain.gbuffer);
        let upscaled = FeaturePlan::resolve(
            &g,
            BackendFacts {
                upscaling: true,
                ..CAPABLE
            },
        );
        assert!(upscaled.temporal && !upscaled.taa_resolve);
        // An upscaler alone still takes the velocity pre-pass.
        let alone = FeaturePlan::resolve(
            &graphics(&[]),
            BackendFacts {
                upscaling: true,
                ..CAPABLE
            },
        );
        assert!(alone.temporal && alone.gbuffer && !alone.taa_resolve);
    }

    #[test]
    fn the_upscaler_fact_is_the_callers_not_the_settings() {
        // The settings request an upscaler, but the caller reports none built.
        let g = graphics(&[Feature::Upscaling]);
        assert!(!FeaturePlan::resolve(&g, CAPABLE).temporal);
    }

    #[test]
    fn the_gbuffer_follows_every_screen_space_consumer() {
        let g = graphics(&[]);
        let p = FeaturePlan::resolve(&g, CAPABLE);
        assert_eq!(
            p.gbuffer,
            p.ssao || p.ssgi || p.temporal || p.ssr_resources,
            "{p:?}"
        );
        let ssr = FeaturePlan::resolve(&graphics(&[Feature::Ssr]), CAPABLE);
        assert!(ssr.ssr && ssr.ssr_resources && ssr.gbuffer);
    }

    #[test]
    fn a_change_follows_wanted_and_built() {
        assert_eq!(FeatureChange::of(true, false), FeatureChange::Build);
        assert_eq!(FeatureChange::of(false, true), FeatureChange::Drop);
        assert_eq!(FeatureChange::of(true, true), FeatureChange::Update);
        assert_eq!(FeatureChange::of(false, false), FeatureChange::Absent);
    }
}

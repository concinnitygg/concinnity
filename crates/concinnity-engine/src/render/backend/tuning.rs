//! The render settings a running world can change.
//!
//! The resolved graphics settings, applied through one plan per change, plus
//! the lights and the fog the editor moves live.

use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::environment::volumetric_fog::FogSettings;
use crate::settings::ResolvedGraphics;
use crate::settings::apply::ApplyPlan;

/// Live render settings: quality, shadows, post-process and lighting.
///
/// All defaulted. The infallible setters default to a no-op, so a backend that
/// reads a setting only at init keeps its init-time value.
/// [`apply_settings`](Self::apply_settings) reports a rebuild it cannot make as
/// [`RenderError::Unsupported`], so the caller learns it did not happen.
pub trait RenderTuning {
    /// Supply the reflection-probe placements: declared `ReflectionProbe`
    /// assets, or a grid the caller auto-seeded over the scene. An empty list
    /// bakes none. The backend bakes a cube per placement and samples the
    /// nearest for the specular reflection.
    /// Pushed once after construction. Default no-op: a backend without probe
    /// support keeps the sky reflection.
    fn set_reflection_probes(&mut self, probes: &[crate::render::probe::capture::ProbePlacement]) {
        let _ = probes;
    }

    /// Apply a settings change: push the live values the plan names, and
    /// reallocate the runtime-switched features it names (TAA, SSAO, SSR,
    /// ray-traced reflections, SSGI, auto-exposure, their G-buffer), whose
    /// pipelines and targets are built from the settings rather than flipped by
    /// a uniform. `graphics` is the full resolved snapshot the plan moves to.
    /// Default: a plan with nothing to reallocate is a no-op, and one with
    /// something is [`RenderError::Unsupported`], for a backend with no live
    /// rebuild path.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] for a plan that reallocates on a
    /// backend with no live rebuild path, or an error if reallocating a feature
    /// fails.
    fn apply_settings(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &ApplyPlan,
    ) -> RenderResult<()> {
        let _ = graphics;
        if plan.rebuild.is_empty() {
            Ok(())
        } else {
            Err(RenderError::Unsupported {
                op: "apply_settings",
            })
        }
    }

    /// Replace the live directional-light set (the sun). Unlike the local
    /// lights, which ride a per-scene storage buffer sized once at init, the
    /// directional slots are a fixed-size array in the shared `LightUniforms`,
    /// so a new set is written in place: the backend re-packs the array and
    /// re-caches whatever it derived from the first light at init (the cascade
    /// shadow direction, the fog sun). Default no-op: a backend that only reads
    /// the lights at init keeps the init-time sun.
    fn update_directional_lights(&mut self, lights: &[crate::components::DirectionalLight]) {
        let _ = lights;
    }

    /// Move the local lights (point, spot and rect area) in place: `lights` is
    /// `lights::build_light_data` over the same lights in the same order, so
    /// the count and every slot assignment are unchanged and only positions
    /// and directions differ. `uniforms` supplies the point array world shader
    /// hooks read; its other fields are ignored. This is how a world that
    /// moves to another simulated frame carries its lights. The spot shadow
    /// slices re-render from their first refresh.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend that cannot move
    /// lights in place, or an error if the light count does not match the
    /// lights built at init.
    fn move_local_lights(
        &mut self,
        lights: &crate::render::light::lights::LightData,
        uniforms: &crate::gfx::render_types::LightUniforms,
    ) -> RenderResult<()> {
        let _ = (lights, uniforms);
        Err(RenderError::Unsupported {
            op: "move_local_lights",
        })
    }

    /// Replace the live volumetric-fog settings, or disable the fog pass when
    /// `None`. Default no-op: the fog pass keeps its init-time settings.
    ///
    /// A backend that built its fog pipeline lazily based on the world's
    /// init-time `VolumetricFog` cannot enable the pass via this call when
    /// the world started with no fog declared; re-enabling fog on a world
    /// that did not declare it at startup requires a relaunch.
    fn update_fog_settings(&mut self, settings: Option<FogSettings>) {
        let _ = settings;
    }
}

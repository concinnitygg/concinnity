//! Runtime replacement of the directional-light set. The lights are packed into
//! `LightUniforms` at init and pushed to the fragment shader every frame, so a
//! new sun is a field rewrite rather than a buffer rebuild. What init derived
//! from the first light and cached -- the cascade shadow direction -- is
//! re-derived here, since nothing else refreshes it.

use concinnity_core::components::DirectionalLight;
use concinnity_core::gfx::render_types::LightUniforms;
use concinnity_core::render::cluster_range::ClusterReach;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::lights::{self, LightData};
use concinnity_core::render::spot_shadow;

use super::context::{MtlContext, write_buffer_slice};

impl MtlContext {
    // Replace the live directional lights. The main pass, fog, raymarch, and RT
    // reflection params all read `light_uniforms` afresh each draw, so they need
    // nothing beyond the rewrite; `shadow.light_dir` is the one init-time cache.
    pub(crate) fn update_directional_lights(&mut self, lights: &[DirectionalLight]) {
        let (directional, num_directional) = lights::directional_light_data(lights);
        if self.light_uniforms.directional == directional
            && self.light_uniforms.num_directional == num_directional
        {
            return;
        }
        self.light_uniforms.directional = directional;
        self.light_uniforms.num_directional = num_directional;
        self.shadow.light_dir = lights::sun_direction(&self.light_uniforms);
    }

    // Move the local lights in place. The light, area and spot tables are
    // single shared-storage buffers the in-flight frames read, so the GPU is
    // drained before they are rewritten; the light-cluster reach, the spot
    // frusta and the spot refresh schedule are re-derived from the new data.
    pub(crate) fn move_local_lights(
        &mut self,
        data: &LightData,
        uniforms: &LightUniforms,
    ) -> RenderResult<()> {
        if data.lights.len() as i32 != self.light_uniforms.num_local_lights
            || data.spot_shadows.len() != self.spot_shadow.count as usize
        {
            return Err(RenderError::Other(
                "move_local_lights: the light count changed since init".into(),
            ));
        }
        self.wait_idle();
        write_buffer_slice(&self.scene.local_light_buffer, &data.lights)?;
        write_buffer_slice(&self.scene.area_light_buffer, &data.area_lights)?;
        write_buffer_slice(&self.spot_shadow.buffer, &data.spot_shadows)?;
        self.scene.cluster_reach = ClusterReach::new(&data.lights);
        self.spot_shadow.frusta = data
            .spot_shadows
            .iter()
            .map(spot_shadow::slice_frustum)
            .collect();
        self.spot_shadow.scheduler = Default::default();
        self.light_uniforms.point = uniforms.point;
        self.light_uniforms.num_point = uniforms.num_point;
        Ok(())
    }
}

//! The lights the DirectX backend shades with, replaced or moved in place.

use concinnity_engine::components;
use concinnity_engine::gfx::render_types::LightUniforms;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{RenderBackend, SceneLights};
use concinnity_engine::render::light::lights;

use super::context::DxContext;

impl SceneLights for DxContext {
    // Replace the live directional lights. The cascade shadow direction and the
    // fog sun are derived from the first light on the CPU, so both are
    // re-derived here. Edge-triggered: an unchanged set touches nothing, and a
    // changed one only re-arms the light CBV ring.
    fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]) {
        self.uniforms.lights.set_directional(lights);
    }

    // Replace the sky-occluding spheres the light block carries; a changed set
    // re-arms the light ring.
    fn update_sky_occluders(
        &mut self,
        occluders: &concinnity_engine::render::light::sky_occluders::SkyOccluders,
    ) {
        self.uniforms.lights.set_sky_occluders(occluders);
    }

    // Move the local lights in place. The light, area and spot tables and the
    // spot slice CBVs are single UPLOAD buffers every in-flight frame reads,
    // so the GPU is drained before they are rewritten; the cluster reach, the
    // spot frusta and the spot refresh schedule are re-derived, and the light
    // CBV ring is re-armed for the point array.
    fn move_local_lights(
        &mut self,
        data: &lights::LightData,
        uniforms: &LightUniforms,
    ) -> RenderResult<()> {
        if !self.uniforms.lights.same_local_count(&data.lights)
            || data.spot_shadows.len() != self.spot_shadow.count() as usize
        {
            return Err(RenderError::Other(
                "move_local_lights: the light count changed since init".into(),
            ));
        }
        self.wait_idle();
        if !data.lights.is_empty() {
            super::draw::upload_static_records(
                &self.uniforms.local_light_buffer,
                &data.lights,
                "local-light",
            )?;
        }
        if !data.area_lights.is_empty() {
            super::draw::upload_static_records(
                &self.scene.area_light.buffer,
                &data.area_lights,
                "area-light",
            )?;
        }
        if !data.spot_shadows.is_empty() {
            super::draw::upload_static_records(
                &self.spot_shadow.buffer,
                &data.spot_shadows,
                "spot-shadow",
            )?;
            super::init::shadow::write_spot_uniforms(
                &self.spot_shadow.ubo,
                &data.spot_shadows,
                self.spot_shadow.ubo_stride,
            )?;
        }
        self.spot_shadow.frusta = data
            .spot_shadows
            .iter()
            .map(concinnity_engine::render::shadow::spot::slice_frustum)
            .collect();
        self.state.planner.shadow.reset_spots();
        self.uniforms.lights.move_local(&data.lights, uniforms);
        Ok(())
    }
}

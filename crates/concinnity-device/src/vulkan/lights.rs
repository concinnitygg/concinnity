//! The lights the Vulkan backend shades with, replaced or moved in place.

use concinnity_engine::components::DirectionalLight;
use concinnity_engine::gfx::render_types::LightUniforms;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{RenderBackend, SceneLights};
use concinnity_engine::render::light::lights::LightData;
use concinnity_engine::render::light::sky_occluders::SkyOccluders;

use super::context::VkContext;

impl SceneLights for VkContext {
    // Replace the live directional lights. The cascade shadow direction and the
    // fog sun are derived from the first light on the CPU, so both are
    // re-derived here. Edge-triggered: an unchanged set touches nothing, and a
    // changed one only re-arms the light UBO ring.
    fn update_directional_lights(&mut self, lights: &[DirectionalLight]) {
        self.uniforms.lights.set_directional(lights);
    }

    // Replace the sky-occluding spheres the light block carries; a changed set
    // re-arms the light ring.
    fn update_sky_occluders(&mut self, occluders: &SkyOccluders) {
        self.uniforms.lights.set_sky_occluders(occluders);
    }

    // Move the local lights in place. The light SSBO, the area table and the
    // spot projections are single host-visible buffers every in-flight frame
    // reads, so the GPU is drained before they are rewritten; the cluster
    // reach and the spot frusta are re-derived, and the light UBO ring is
    // re-armed for the point array.
    fn move_local_lights(
        &mut self,
        data: &LightData,
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
        super::draw::upload_static_records(&self.uniforms.local_light_buffer, &data.lights);
        super::draw::upload_static_records(&self.area_light.buffer, &data.area_lights);
        self.spot_shadow.rewrite(&data.spot_shadows);
        self.state.planner.shadow.reset_spots();
        self.uniforms.lights.move_local(&data.lights, uniforms);
        Ok(())
    }
}

//! Runtime replacement of the directional-light set. The lights are packed into
//! `LightUniforms` at init and pushed to the fragment shader every frame, so a
//! new sun is a field rewrite rather than a buffer rebuild. What init derived
//! from the first light and cached -- the cascade shadow direction -- is
//! re-derived here, since nothing else refreshes it.

use concinnity_core::components::DirectionalLight;
use concinnity_core::gfx::render_types::LightUniforms;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::light::lights::LightData;
use concinnity_core::render::shadow::spot;

use super::context::{MtlContext, write_buffer_slice};

impl MtlContext {
    // Replace the live directional lights. Every reader, the shadow cascades
    // included, reads the sun from the light state each frame.
    pub(crate) fn update_directional_lights(&mut self, lights: &[DirectionalLight]) {
        self.lights.set_directional(lights);
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
        if !self.lights.same_local_count(&data.lights)
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
        self.spot_shadow.frusta = data.spot_shadows.iter().map(spot::slice_frustum).collect();
        self.state.planner.shadow.reset_spots();
        self.lights.move_local(&data.lights, uniforms);
        Ok(())
    }
}

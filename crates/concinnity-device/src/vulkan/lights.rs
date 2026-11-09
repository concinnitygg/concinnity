//! The per-light tables Vulkan's passes read: the local-light SSBO, the area
//! table and the spot projections, single host-visible buffers rewritten in
//! place when the lights move.

use concinnity_engine::render::backend::LightTables;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::light::lights::LightData;

use super::context::VkContext;

impl LightTables for VkContext {
    fn spot_slice_count(&self) -> usize {
        self.spot_shadow.count() as usize
    }

    fn rewrite_light_tables(&mut self, data: &LightData) -> RenderResult<()> {
        super::draw::upload_static_records(&self.uniforms.local_light_buffer, &data.lights);
        super::draw::upload_static_records(&self.area_light.buffer, &data.area_lights);
        self.spot_shadow.rewrite(&data.spot_shadows);
        Ok(())
    }
}

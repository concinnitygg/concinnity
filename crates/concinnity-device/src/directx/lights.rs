//! The per-light tables DirectX's passes read: the local-light, area-light and
//! spot-shadow tables and the spot slice CBVs, single UPLOAD buffers rewritten
//! in place when the lights move.

use concinnity_engine::render::backend::LightTables;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::light::lights::LightData;

use super::context::DxContext;

impl LightTables for DxContext {
    fn spot_slice_count(&self) -> usize {
        self.spot_shadow.count() as usize
    }

    fn rewrite_light_tables(&mut self, data: &LightData) -> RenderResult<()> {
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
        Ok(())
    }
}

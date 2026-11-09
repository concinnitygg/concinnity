//! The per-light tables Metal's passes read: the local-light, area-light and
//! spot-shadow buffers, single shared-storage buffers rewritten in place when
//! the lights move.

use concinnity_engine::render::backend::LightTables;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::light::lights::LightData;
use concinnity_engine::render::shadow::spot;

use super::context::{MtlContext, write_buffer_slice};

impl LightTables for MtlContext {
    fn spot_slice_count(&self) -> usize {
        self.spot_shadow.count as usize
    }

    fn rewrite_light_tables(&mut self, data: &LightData) -> RenderResult<()> {
        write_buffer_slice(&self.scene.local_light_buffer, &data.lights)?;
        write_buffer_slice(&self.scene.area_light_buffer, &data.area_lights)?;
        write_buffer_slice(&self.spot_shadow.buffer, &data.spot_shadows)?;
        self.spot_shadow.frusta = data.spot_shadows.iter().map(spot::slice_frustum).collect();
        Ok(())
    }
}

//! The material parameter table on Metal: core's table ringed over the same
//! slots as the object buffer (the frames in flight plus the probe bake's), each
//! slot's copy rewritten only when it is fresh or behind a change.

use concinnity_core::gfx::render_types::{GpuMaterialParams, MATERIAL_PARAM_COUNT};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::material_params::MaterialParamTable;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use super::context::{bytes_of_slice, write_buffer_region};
use super::frame_rings::TransientRing;

// The Metal buffer index the main pass reads the table from, on both stages.
pub(super) const MATERIAL_PARAMS_BUFFER_INDEX: usize = 16;

pub(super) struct MaterialParamRing {
    table: MaterialParamTable,
    ring: TransientRing,
}

impl MaterialParamRing {
    // `depth` ring slots over `rows` (see `material_params::rows`).
    pub(super) fn new(rows: Vec<GpuMaterialParams>, depth: usize) -> Self {
        let depth = depth.max(1);
        Self {
            table: MaterialParamTable::new(rows, depth),
            ring: TransientRing::new(depth),
        }
    }

    pub(super) fn set(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) -> bool {
        self.table.set(row, params)
    }

    // `slot`'s copy of the table, written first when it is new or out of date.
    pub(super) fn buffer(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let (buf, fresh) = self.ring.slot_fresh(device, slot, self.table.byte_len())?;
        let behind = self.table.take_upload(slot).is_some();
        if fresh || behind {
            write_buffer_region(&buf, 0, bytes_of_slice(self.table.rows()))?;
        }
        Ok(buf)
    }
}

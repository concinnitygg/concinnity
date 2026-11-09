//! The material parameter table on Metal: the scene's table ringed over the same
//! slots as the object buffer (the frames in flight plus the probe bake's), each
//! slot's copy rewritten only when it is fresh or behind a change.

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::material_params::MaterialParamTable;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use super::context::{bytes_of_slice, write_buffer_region};
use super::frame_rings::TransientRing;

// The Metal buffer index the main pass reads the table from, on both stages.
pub(super) const MATERIAL_PARAMS_BUFFER_INDEX: usize = 16;

pub(super) struct MaterialParamRing {
    ring: TransientRing,
}

impl MaterialParamRing {
    // One ring slot per copy `table` tracks.
    pub(super) fn new(table: &MaterialParamTable) -> Self {
        Self {
            ring: TransientRing::new(table.copies().max(1)),
        }
    }

    // `slot`'s copy of `table`, written first when it is new or out of date.
    pub(super) fn buffer(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        table: &mut MaterialParamTable,
        slot: usize,
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let (buf, fresh) = self.ring.slot_fresh(device, slot, table.byte_len())?;
        let behind = table.take_upload(slot).is_some();
        if fresh || behind {
            write_buffer_region(&buf, 0, bytes_of_slice(table.rows()))?;
        }
        Ok(buf)
    }
}

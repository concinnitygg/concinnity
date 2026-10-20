//! The far field's view tables on Metal: one shared buffer per version of the
//! tables, replaced rather than rewritten when they change, so a frame in
//! flight keeps reading the one it bound.

use concinnity_core::render::backend::error::RenderResult;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

// The Metal buffer index the main pass's fragment stage reads the tables from.
pub(super) const CANOPY_TABLES_BUFFER_INDEX: usize = 22;

#[derive(Default)]
pub(super) struct CanopyTables {
    rows: Vec<[f32; 4]>,
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl CanopyTables {
    // Take `rows` as the tables, uploading a new buffer when they changed.
    pub(super) fn update(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        rows: &[[f32; 4]],
    ) -> RenderResult<()> {
        if self.buffer.is_some() && self.rows == rows {
            return Ok(());
        }
        self.buffer = Some(super::raytrace::upload_buffer(
            device,
            rows,
            "canopy tables",
        )?);
        self.rows.clear();
        self.rows.extend_from_slice(rows);
        Ok(())
    }

    pub(super) fn buffer(&self) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.buffer.as_ref()
    }
}

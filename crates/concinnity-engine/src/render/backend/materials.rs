//! The material parameter table world Shaders read, rewritten a row at a time
//! while the world runs.

use crate::gfx::render_types::MATERIAL_PARAM_COUNT;

/// The material parameter table's rows.
pub trait MaterialParamRows {
    /// Replace one row of the material parameter table with a material's new
    /// `params`; `row` is [`crate::render::material_params::row_of`] of its
    /// handle. Every draw reading the row picks the values up on the next
    /// frame. Driven by the editor's live material seam. Default no-op, for a
    /// backend that draws no world Shader.
    fn set_material_params(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        let _ = (row, params);
    }
}

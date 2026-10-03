//! The material parameter table the main pass reads a Shader's `params` from:
//! an all-zero row 0 for draws without a material, then one row per material in
//! `MaterialHandle` order. Each object record carries its material's row in
//! `params_index`, so draws sharing a material share its row, and a change to a
//! material's parameters rewrites one row without touching any object.
//!
//! The CPU owns the rows. A backend rings the table over its frames in flight
//! and rewrites a frame's copy only after a row changed.

use alloc::vec::Vec;

use crate::ecs::MaterialHandle;
use crate::gfx::render_types::{GpuMaterialParams, MATERIAL_PARAM_COUNT};
use crate::render::frame_dirty::FrameDirty;

/// The row a draw without a material reads: all zeros.
pub const NO_MATERIAL_ROW: u32 = 0;

/// The row `material`'s parameters live in.
pub const fn row_of(material: Option<MaterialHandle>) -> u32 {
    match material {
        Some(handle) => handle.0 + 1,
        None => NO_MATERIAL_ROW,
    }
}

/// The table's rows for every material's `params`, given in handle order.
pub fn rows(
    params: impl IntoIterator<Item = [f32; MATERIAL_PARAM_COUNT]>,
) -> Vec<GpuMaterialParams> {
    core::iter::once(GpuMaterialParams::default())
        .chain(
            params
                .into_iter()
                .map(|values| GpuMaterialParams { values }),
        )
        .collect()
}

/// The table plus which frame-in-flight copies of it still need a write.
#[derive(Clone, Debug)]
pub struct MaterialParamTable {
    rows: Vec<GpuMaterialParams>,
    dirty: FrameDirty,
}

impl MaterialParamTable {
    /// A table over `rows` (see [`rows`]) ringed over `frames` copies, each
    /// pending so the ring seeds itself. An empty `rows` still gets the zero
    /// row.
    pub fn new(mut rows: Vec<GpuMaterialParams>, frames: usize) -> Self {
        if rows.is_empty() {
            rows.push(GpuMaterialParams::default());
        }
        Self {
            rows,
            dirty: FrameDirty::new(frames),
        }
    }

    /// Every row, the zero row first.
    pub fn rows(&self) -> &[GpuMaterialParams] {
        &self.rows
    }

    /// The table's size in bytes, which a copy of it is allocated at.
    pub fn byte_len(&self) -> usize {
        core::mem::size_of_val(self.rows.as_slice())
    }

    /// Replace row `row` with `values`. False when the row is the zero row or
    /// past the table, which leaves the table untouched. A change marks every
    /// frame's copy pending; writing the values a row already holds does not.
    pub fn set(&mut self, row: u32, values: [f32; MATERIAL_PARAM_COUNT]) -> bool {
        if row == NO_MATERIAL_ROW {
            return false;
        }
        let Some(slot) = self.rows.get_mut(row as usize) else {
            return false;
        };
        if slot.values != values {
            slot.values = values;
            self.dirty.mark_all();
        }
        true
    }

    /// The rows to write into `frame`'s copy, or `None` when that copy is
    /// already current.
    pub fn take_upload(&mut self, frame: usize) -> Option<&[GpuMaterialParams]> {
        self.dirty.take(frame).then_some(self.rows.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(first: f32) -> [f32; MATERIAL_PARAM_COUNT] {
        core::array::from_fn(|i| first + i as f32)
    }

    #[test]
    fn a_material_reads_the_row_after_its_handle() {
        assert_eq!(row_of(None), NO_MATERIAL_ROW);
        assert_eq!(row_of(Some(MaterialHandle::new(0))), 1);
        assert_eq!(row_of(Some(MaterialHandle::new(6))), 7);

        let rows = rows([params(10.0), params(20.0)]);
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[row_of(Some(MaterialHandle::new(0))) as usize].values,
            params(10.0)
        );
        assert_eq!(
            rows[row_of(Some(MaterialHandle::new(1))) as usize].values,
            params(20.0)
        );
    }

    #[test]
    fn row_zero_is_all_zeros_and_cannot_be_written() {
        let mut table = MaterialParamTable::new(rows([params(1.0)]), 2);
        assert_eq!(table.rows()[0], GpuMaterialParams::default());
        assert!(!table.set(NO_MATERIAL_ROW, params(5.0)));
        assert_eq!(table.rows()[0], GpuMaterialParams::default());
    }

    #[test]
    fn a_world_without_materials_still_has_the_zero_row() {
        let table = MaterialParamTable::new(rows([]), 2);
        assert_eq!(table.rows(), &[GpuMaterialParams::default()]);
        assert_eq!(table.byte_len(), 32);
        assert_eq!(MaterialParamTable::new(Vec::new(), 2).rows().len(), 1);
    }

    // Draws are not rows: however many objects share a material, the table
    // holds it once, and every one of them reads the same row.
    #[test]
    fn draws_sharing_a_material_share_its_row() {
        let draws = [
            Some(MaterialHandle::new(1)),
            None,
            Some(MaterialHandle::new(1)),
            Some(MaterialHandle::new(0)),
        ];
        let table = MaterialParamTable::new(rows([params(1.0), params(2.0)]), 1);
        assert_eq!(table.rows().len(), 3, "zero row plus one per material");
        let read: Vec<u32> = draws.iter().map(|&m| row_of(m)).collect();
        assert_eq!(read, [2, 0, 2, 1]);
        assert_eq!(table.rows()[read[0] as usize].values, params(2.0));
    }

    #[test]
    fn every_frame_uploads_once_until_a_row_changes() {
        let mut table = MaterialParamTable::new(rows([params(1.0)]), 2);
        assert!(table.take_upload(0).is_some());
        assert!(table.take_upload(1).is_some());
        assert!(table.take_upload(0).is_none());
        assert!(table.take_upload(1).is_none());

        assert!(table.set(1, params(9.0)));
        let uploaded = table.take_upload(1).expect("a change re-arms every frame");
        assert_eq!(uploaded[1].values, params(9.0));
        assert!(table.take_upload(0).is_some());
        assert!(table.take_upload(0).is_none());
    }

    #[test]
    fn rewriting_a_row_with_its_own_values_uploads_nothing() {
        let mut table = MaterialParamTable::new(rows([params(1.0)]), 1);
        table.take_upload(0);
        assert!(table.set(1, params(1.0)));
        assert!(table.take_upload(0).is_none());
    }

    #[test]
    fn a_row_past_the_table_is_refused() {
        let mut table = MaterialParamTable::new(rows([params(1.0)]), 1);
        table.take_upload(0);
        assert!(!table.set(2, params(3.0)));
        assert!(table.take_upload(0).is_none());
    }
}

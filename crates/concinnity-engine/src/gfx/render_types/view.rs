//! The view sampling pass's layouts: a coarse grid over the frame's depth,
//! each cell's sample and nearest texel, read back for the CPU to measure what
//! the camera saw.

use crate::render::depth::{DEPTH_FAR, DEPTH_NEAR};
use concinnity_core::math::SaturatingFrom;

/// The most cells the view sampling grid holds, and so the readback's length.
pub const VIEW_SAMPLE_CELLS: usize = 1024;

// The sampling keeps each cell's nearest texel by an atomic max over the
// depth's bits, which holds only while nearer depths are greater and none is
// negative.
const _: () = assert!(DEPTH_NEAR > DEPTH_FAR && DEPTH_FAR >= 0.0);

/// Threads across each side of the group that samples one cell.
pub const VIEW_SAMPLE_GROUP: u32 = 16;

/// How the view sampling pass cuts a depth into square cells.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ViewSampleGrid {
    /// Each cell's edge, in texels.
    pub cell: u32,
    /// Cells across.
    pub columns: u32,
    /// Cells down.
    pub rows: u32,
}

impl ViewSampleGrid {
    /// The grid over a `width` by `height` depth with the smallest square
    /// cells that keep it within [`VIEW_SAMPLE_CELLS`].
    pub fn over(width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let per_cell = (u64::from(width) * u64::from(height)).div_ceil(VIEW_SAMPLE_CELLS as u64);
        let root = per_cell.isqrt();
        let mut cell = u32::try_from(root + u64::from(root * root < per_cell)).unwrap_or(u32::MAX);
        loop {
            let grid = Self {
                cell,
                columns: width.div_ceil(cell),
                rows: height.div_ceil(cell),
            };
            if grid.cells() <= VIEW_SAMPLE_CELLS {
                return grid;
            }
            cell += 1;
        }
    }

    /// How many cells the grid holds.
    pub fn cells(&self) -> usize {
        self.columns as usize * self.rows as usize
    }

    /// Where in every cell measurement `serial` samples the depth: a texel
    /// offset from each cell's corner, moved each measurement along a
    /// low-discrepancy sequence, so a band thinner than a cell is seen in
    /// its share of the measurements.
    pub fn spot(&self, serial: u32) -> [u32; 2] {
        // The plastic number's R2 sequence, its steps as 32-bit fractions.
        const STEPS: [u32; 2] = [3_242_174_889, 2_447_445_413];
        STEPS.map(|step| {
            let fraction = (1_u32 << 31).wrapping_add(serial.wrapping_mul(step));
            u32::saturating_from((u64::from(fraction) * u64::from(self.cell)) >> 32)
        })
    }

    /// The texel cell `index` samples at `spot` ([`Self::spot`]) in a
    /// `width` by `height` depth.
    pub fn sample(&self, index: usize, spot: [u32; 2], width: u32, height: u32) -> [u32; 2] {
        let index = u32::saturating_from(index);
        let column = index % self.columns.max(1);
        let row = index / self.columns.max(1);
        [
            (column * self.cell + spot[0]).min(width.saturating_sub(1)),
            (row * self.cell + spot[1]).min(height.saturating_sub(1)),
        ]
    }
}

/// What the view sampling pass reads. 32 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct ViewSampleParams {
    /// The depth's width in texels.
    pub width: u32,
    /// The depth's height in texels.
    pub height: u32,
    /// Each cell's edge, in texels.
    pub cell: u32,
    /// Cells across.
    pub columns: u32,
    /// Cells down.
    pub rows: u32,
    /// The measurement's number, written into every cell.
    pub serial: u32,
    /// Where in every cell the depth is sampled, from its corner
    /// ([`ViewSampleGrid::spot`]).
    pub spot: [u32; 2],
}

impl ViewSampleParams {
    /// The parameters that sample a `width` by `height` depth over `grid`,
    /// numbered `serial`.
    pub fn new(width: u32, height: u32, grid: ViewSampleGrid, serial: u32) -> Self {
        Self {
            width,
            height,
            cell: grid.cell,
            columns: grid.columns,
            rows: grid.rows,
            serial,
            spot: grid.spot(serial),
        }
    }
}

/// One cell's samples of the depth, as the pass writes them. 16 bytes.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ViewSampleCell {
    /// The device depth at the cell's sample texel
    /// ([`ViewSampleParams::spot`]).
    pub depth: f32,
    /// The device depth of the cell's nearest texel; the cleared value when
    /// no surface reached the cell.
    pub nearest: f32,
    /// Where the nearest texel lies, packed `(y << 16) | x`.
    pub texel: u32,
    /// The measurement's number, from its [`ViewSampleParams`]; 0 in a cell
    /// no measurement wrote.
    pub serial: u32,
}

impl ViewSampleCell {
    /// The nearest texel's position.
    pub fn nearest_texel(&self) -> [u32; 2] {
        [self.texel & 0xffff, self.texel >> 16]
    }
}

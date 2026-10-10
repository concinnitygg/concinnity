//! The grass pass's blocks: the per-frame parameters every grass stage reads
//! and the layers inside them, one visible blade as the kernel appends it, and
//! the draw arguments the kernel fills. Match `GrassParams` / `GrassLayer` /
//! `GrassBlade` in `shaders/grass.hlsl`.

/// Vertices in one blade strip: seven pairs up the blade and the tip.
pub const GRASS_BLADE_VERTICES: u32 = 15;

/// Draw-argument slots the args buffer holds. The kernel fills one each frame
/// and resets the other for the next, so no separate clear runs.
pub const GRASS_ARGS_SLOTS: usize = 2;

/// Bytes in one slot of non-indexed draw arguments, the layout Metal, DirectX
/// and Vulkan share: vertex count, instance count, first vertex, first
/// instance.
pub const GRASS_ARGS_STRIDE: usize = 16;

/// Bytes in the whole args buffer.
pub const GRASS_ARGS_BYTES: usize = GRASS_ARGS_SLOTS * GRASS_ARGS_STRIDE;

/// Most grass layers one frame grows: the layers within reach of the camera,
/// in the order the world declares them.
pub const MAX_GRASS_LAYERS: usize = 16;

/// One grass layer as a frame's kernel and draws read it: the terrain it roots
/// in, the block of tiles this frame covers on it, and the look of its blades.
/// 128 bytes, every member in its own 16-byte row.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GrassLayerGpu {
    /// The terrain's ground rectangle: min x, min z, max x, max z.
    pub rect: [f32; 4],
    /// Tile coordinates of the first tile this frame covers.
    pub tile_origin: [i32; 2],
    /// Tiles covered along x and z.
    pub tile_count: [u32; 2],
    /// World height of the terrain's base.
    pub base_y: f32,
    /// Lowest ground on the terrain, world height.
    pub min_y: f32,
    /// Highest ground on the terrain, world height.
    pub max_y: f32,
    /// Grid cells along each side of the terrain.
    pub grid_resolution: u32,
    /// Index of the terrain's first height in the heights buffer.
    pub heights_offset: u32,
    /// Index of the mask's first texel in the mask buffer.
    pub mask_offset: u32,
    /// Mask width in the low 16 bits, height in the high 16; 0 for no mask.
    pub mask_size: u32,
    /// Salts the layer's placement hashes, so layers sharing a terrain grow
    /// their own blades.
    pub seed: u32,
    /// Mean blade height, in meters.
    pub height: f32,
    /// Height variation as a fraction of `height`.
    pub height_variance: f32,
    /// Blade width at the root, in meters.
    pub width: f32,
    /// Clump diameter, in meters.
    pub clump_size: f32,
    /// Resistance to bending, in [0, 1].
    pub stiffness: f32,
    /// Hue and brightness variation, in [0, 1].
    pub color_variation: f32,
    /// Edge of one cell, in meters.
    pub cell_size: f32,
    /// Cells along each tile edge.
    pub cells_per_side: u32,
    /// Linear RGB at the root; `w` unused.
    pub root_color: [f32; 4],
    /// Linear RGB at the tip; `w` unused.
    pub tip_color: [f32; 4],
}

/// Per-frame grass parameters: the camera, the wind, and the layers in reach.
/// Every member a `float4` row, a run of four scalars, or a whole layer, so
/// the constant-buffer and storage layouts agree on every target.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GrassParams {
    /// Camera position the kernel measures distance from.
    pub cam_pos: [f32; 3],
    /// Distance past which no blade is kept.
    pub draw_distance: f32,
    /// The camera's world-space frustum planes, `(normal, d)` with
    /// `dot(normal, p) + d >= 0` inside.
    pub frustum: [[f32; 4]; 6],
    /// Wind direction `[x, z]`, speed and gustiness (see `WindField::gpu_rows`).
    pub wind: [f32; 4],
    /// Inverse gust width; `yzw` unused.
    pub wind_gust: [f32; 4],
    /// Entries of `layers` in use.
    pub layer_count: u32,
    /// Blades the visible-blade buffer holds.
    pub capacity: u32,
    /// The draw-argument slot this frame fills; the other one is reset.
    pub args_slot: u32,
    /// Edge of one tile, in meters.
    pub tile_size: f32,
    /// The layers this frame grows.
    pub layers: [GrassLayerGpu; MAX_GRASS_LAYERS],
}

/// One visible blade, as the kernel appends it and the draws read it.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GpuGrassBlade {
    /// Root position (xyz) and facing angle in radians (w).
    pub root_facing: [f32; 4],
    /// Height and root width (as `f32` bit patterns), the static lean `[x, z]`
    /// as two halves, and the blade's bits: the clump's and the blade's random
    /// bits and the layer it belongs to.
    pub shape: [u32; 4],
}

/// The initial contents of the args buffer: every slot a draw of one blade
/// strip with no instances.
pub fn initial_grass_args() -> [u32; GRASS_ARGS_SLOTS * 4] {
    let mut args = [0u32; GRASS_ARGS_SLOTS * 4];
    for slot in args.chunks_exact_mut(4) {
        slot[0] = GRASS_BLADE_VERTICES;
    }
    args
}

/// Byte offset of draw-argument slot `slot`.
pub fn grass_args_offset(slot: u32) -> usize {
    (slot as usize % GRASS_ARGS_SLOTS) * GRASS_ARGS_STRIDE
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn grass_params_layout_matches_msl() {
        assert_eq!(size_of::<GrassParams>(), 160 + 128 * MAX_GRASS_LAYERS);
        assert_eq!(offset_of!(GrassParams, cam_pos), 0);
        assert_eq!(offset_of!(GrassParams, draw_distance), 12);
        assert_eq!(offset_of!(GrassParams, frustum), 16);
        assert_eq!(offset_of!(GrassParams, wind), 112);
        assert_eq!(offset_of!(GrassParams, wind_gust), 128);
        assert_eq!(offset_of!(GrassParams, layer_count), 144);
        assert_eq!(offset_of!(GrassParams, capacity), 148);
        assert_eq!(offset_of!(GrassParams, args_slot), 152);
        assert_eq!(offset_of!(GrassParams, tile_size), 156);
        assert_eq!(offset_of!(GrassParams, layers), 160);
    }

    #[test]
    fn grass_layer_layout_matches_msl() {
        assert_eq!(size_of::<GrassLayerGpu>(), 128);
        assert_eq!(offset_of!(GrassLayerGpu, rect), 0);
        assert_eq!(offset_of!(GrassLayerGpu, tile_origin), 16);
        assert_eq!(offset_of!(GrassLayerGpu, tile_count), 24);
        assert_eq!(offset_of!(GrassLayerGpu, base_y), 32);
        assert_eq!(offset_of!(GrassLayerGpu, min_y), 36);
        assert_eq!(offset_of!(GrassLayerGpu, max_y), 40);
        assert_eq!(offset_of!(GrassLayerGpu, grid_resolution), 44);
        assert_eq!(offset_of!(GrassLayerGpu, heights_offset), 48);
        assert_eq!(offset_of!(GrassLayerGpu, mask_offset), 52);
        assert_eq!(offset_of!(GrassLayerGpu, mask_size), 56);
        assert_eq!(offset_of!(GrassLayerGpu, seed), 60);
        assert_eq!(offset_of!(GrassLayerGpu, height), 64);
        assert_eq!(offset_of!(GrassLayerGpu, height_variance), 68);
        assert_eq!(offset_of!(GrassLayerGpu, width), 72);
        assert_eq!(offset_of!(GrassLayerGpu, clump_size), 76);
        assert_eq!(offset_of!(GrassLayerGpu, stiffness), 80);
        assert_eq!(offset_of!(GrassLayerGpu, color_variation), 84);
        assert_eq!(offset_of!(GrassLayerGpu, cell_size), 88);
        assert_eq!(offset_of!(GrassLayerGpu, cells_per_side), 92);
        assert_eq!(offset_of!(GrassLayerGpu, root_color), 96);
        assert_eq!(offset_of!(GrassLayerGpu, tip_color), 112);
    }

    #[test]
    fn grass_blade_layout_matches_msl() {
        assert_eq!(size_of::<GpuGrassBlade>(), 32);
        assert_eq!(offset_of!(GpuGrassBlade, root_facing), 0);
        assert_eq!(offset_of!(GpuGrassBlade, shape), 16);
    }

    #[test]
    fn every_args_slot_starts_as_an_empty_strip_draw() {
        let args = initial_grass_args();
        assert_eq!(args, [15, 0, 0, 0, 15, 0, 0, 0]);
        assert_eq!(core::mem::size_of_val(&args), GRASS_ARGS_BYTES);
    }

    #[test]
    fn the_args_offset_alternates_between_the_two_slots() {
        assert_eq!(grass_args_offset(0), 0);
        assert_eq!(grass_args_offset(1), 16);
        assert_eq!(grass_args_offset(2), 0);
    }
}

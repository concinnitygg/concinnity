//! The grass pass's blocks: the per-frame parameters every grass stage reads
//! and the layers inside them, one visible blade as the kernel appends it, the
//! draw arguments the kernel fills, and the bend pass's parameters. Match
//! `GrassParams` / `GrassLayer` / `GrassBlade` / `GrassBendParams` in
//! `shaders/grass.hlsl`.

use crate::render::grass::lod::{GRASS_LOD_COUNT, GRASS_LOD_VERTEX_STRIDE, GRASS_LOD_VERTICES};

/// Draw-argument slots the args buffer holds. The kernel fills one each frame
/// and resets the other for the next, so no separate clear runs.
pub const GRASS_ARGS_SLOTS: usize = 2;

/// Bytes in one record of non-indexed draw arguments, the layout Metal,
/// DirectX and Vulkan share: vertex count, instance count, first vertex, first
/// instance.
pub const GRASS_ARGS_STRIDE: usize = 16;

/// Bytes in one slot: one draw per detail level, coarsening in order.
pub const GRASS_ARGS_SLOT_BYTES: usize = GRASS_LOD_COUNT * GRASS_ARGS_STRIDE;

/// Bytes in the whole args buffer.
pub const GRASS_ARGS_BYTES: usize = GRASS_ARGS_SLOTS * GRASS_ARGS_SLOT_BYTES;

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
    /// Linear RGB at the root.
    pub root_color: [f32; 3],
    /// How far, in meters, a blade can reach from its root in any direction:
    /// its tallest height, bent any way, plus half its widest width.
    pub reach: f32,
    /// Linear RGB at the tip; `w` unused.
    pub tip_color: [f32; 4],
}

/// Per-frame grass parameters: the camera, the wind, the occlusion test, the
/// detail levels, and the layers in reach. Every member a `float4` row, a run
/// of four scalars, or a whole layer, so the constant-buffer and storage
/// layouts agree on every target.
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
    /// The previous frame's view-projection, which the Hi-Z pyramid was built
    /// through, column-major.
    pub prev_vp: [[f32; 4]; 4],
    /// Pyramid base size in texels.
    pub hiz_size: [f32; 2],
    /// Pyramid mip levels.
    pub hiz_mip_count: u32,
    /// 1 when the pyramid holds last frame's depth and tiles are tested
    /// against it.
    pub hiz_enabled: u32,
    /// The first blade of each detail level's region; `w` unused.
    pub lod_base: [u32; 4],
    /// Blades each level's region holds; `w` unused.
    pub lod_capacity: [u32; 4],
    /// Where the second and third levels start, the morph fraction, and the
    /// distance thinning starts at.
    pub lod_distances: [f32; 4],
    /// The fewest blades thinning keeps, the shrink band, where the fade
    /// starts, and the widest a blade grows.
    pub thinning: [f32; 4],
    /// Entries of `layers` in use.
    pub layer_count: u32,
    /// The draw-argument slot this frame fills; the other one is reset.
    pub args_slot: u32,
    /// Edge of one tile, in meters.
    pub tile_size: f32,
    /// Distance from the camera past which the kernel places no blade: the
    /// draw distance for the view, nearer for the shadow cascade.
    pub cull_distance: f32,
    /// The bend field's window: the cell at its corner this frame (`xy`) and
    /// last frame (`zw`), in field cells.
    pub bend_window: [i32; 4],
    /// Edge of one bend field cell, in meters.
    pub bend_cell_size: f32,
    /// Bend field cells along each side of the window.
    pub bend_resolution: u32,
    /// The half of the bend buffer this frame's field is in; last frame's is
    /// the other.
    pub bend_half: u32,
    /// 1 when the other half holds last frame's field.
    pub bend_prev_valid: u32,
    /// The light view-projection the shadow draw projects through,
    /// column-major; unused outside the shadow block.
    pub shadow_vp: [[f32; 4]; 4],
    /// The direction toward the light (`xyz`) and the clock the shadow draw
    /// sways the blades at (`w`); unused outside the shadow block.
    pub shadow_light: [f32; 4],
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
    /// How far it is trampled, `[x, z]` as two halves, this frame (`x`) and
    /// last frame (`y`); `zw` unused.
    pub bend: [u32; 4],
}

/// Most footprints one frame stamps into the bend field.
pub const MAX_GRASS_STAMPS: usize = 64;

/// The bend pass's parameters: where the field's window lies now and lay last
/// frame, how far it relaxes this frame, and the footprints it stamps.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GrassBendParams {
    /// The cell at the window's corner this frame (`xy`) and last frame (`zw`).
    pub window: [i32; 4],
    /// Cells along each side of the window.
    pub resolution: u32,
    /// Edge of one cell, in meters.
    pub cell_size: f32,
    /// What last frame's bend is scaled by as it springs back.
    pub decay: f32,
    /// Entries of `stamps` in use.
    pub stamp_count: u32,
    /// The half of the buffer this frame writes; it reads the other.
    pub write_half: u32,
    /// 1 when the other half holds last frame's field.
    pub prev_valid: u32,
    /// Keeps the stamps on a 16-byte row.
    pub _pad: [u32; 2],
    /// Each footprint: center `x`, `z`, radius and strength.
    pub stamps: [[f32; 4]; MAX_GRASS_STAMPS],
}

/// The initial contents of the args buffer: every record a draw of its
/// level's strip with no instances, starting at the level's vertex ids.
pub fn initial_grass_args() -> [u32; GRASS_ARGS_BYTES / 4] {
    let mut args = [0u32; GRASS_ARGS_BYTES / 4];
    for (i, record) in args.chunks_exact_mut(4).enumerate() {
        let lod = i % GRASS_LOD_COUNT;
        record[0] = GRASS_LOD_VERTICES[lod];
        record[2] = lod as u32 * GRASS_LOD_VERTEX_STRIDE;
    }
    args
}

/// Byte offset of draw-argument slot `slot`: its levels' records follow it,
/// [`GRASS_ARGS_STRIDE`] apart.
pub fn grass_args_offset(slot: u32) -> usize {
    (slot as usize % GRASS_ARGS_SLOTS) * GRASS_ARGS_SLOT_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn grass_params_layout_matches_msl() {
        assert_eq!(size_of::<GrassParams>(), 416 + 128 * MAX_GRASS_LAYERS);
        assert_eq!(offset_of!(GrassParams, cam_pos), 0);
        assert_eq!(offset_of!(GrassParams, draw_distance), 12);
        assert_eq!(offset_of!(GrassParams, frustum), 16);
        assert_eq!(offset_of!(GrassParams, wind), 112);
        assert_eq!(offset_of!(GrassParams, wind_gust), 128);
        assert_eq!(offset_of!(GrassParams, prev_vp), 144);
        assert_eq!(offset_of!(GrassParams, hiz_size), 208);
        assert_eq!(offset_of!(GrassParams, hiz_mip_count), 216);
        assert_eq!(offset_of!(GrassParams, hiz_enabled), 220);
        assert_eq!(offset_of!(GrassParams, lod_base), 224);
        assert_eq!(offset_of!(GrassParams, lod_capacity), 240);
        assert_eq!(offset_of!(GrassParams, lod_distances), 256);
        assert_eq!(offset_of!(GrassParams, thinning), 272);
        assert_eq!(offset_of!(GrassParams, layer_count), 288);
        assert_eq!(offset_of!(GrassParams, args_slot), 292);
        assert_eq!(offset_of!(GrassParams, tile_size), 296);
        assert_eq!(offset_of!(GrassParams, cull_distance), 300);
        assert_eq!(offset_of!(GrassParams, bend_window), 304);
        assert_eq!(offset_of!(GrassParams, bend_cell_size), 320);
        assert_eq!(offset_of!(GrassParams, bend_resolution), 324);
        assert_eq!(offset_of!(GrassParams, bend_half), 328);
        assert_eq!(offset_of!(GrassParams, bend_prev_valid), 332);
        assert_eq!(offset_of!(GrassParams, shadow_vp), 336);
        assert_eq!(offset_of!(GrassParams, shadow_light), 400);
        assert_eq!(offset_of!(GrassParams, layers), 416);
        // Metal sets the block inline, which caps it at 4 KiB.
        assert!(size_of::<GrassParams>() <= 4096);
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
        assert_eq!(offset_of!(GrassLayerGpu, reach), 108);
        assert_eq!(offset_of!(GrassLayerGpu, tip_color), 112);
    }

    #[test]
    fn grass_blade_layout_matches_msl() {
        assert_eq!(size_of::<GpuGrassBlade>(), 48);
        assert_eq!(offset_of!(GpuGrassBlade, root_facing), 0);
        assert_eq!(offset_of!(GpuGrassBlade, shape), 16);
        assert_eq!(offset_of!(GpuGrassBlade, bend), 32);
    }

    #[test]
    fn grass_bend_params_layout_matches_msl() {
        assert_eq!(size_of::<GrassBendParams>(), 48 + 16 * MAX_GRASS_STAMPS);
        assert_eq!(offset_of!(GrassBendParams, window), 0);
        assert_eq!(offset_of!(GrassBendParams, resolution), 16);
        assert_eq!(offset_of!(GrassBendParams, cell_size), 20);
        assert_eq!(offset_of!(GrassBendParams, decay), 24);
        assert_eq!(offset_of!(GrassBendParams, stamp_count), 28);
        assert_eq!(offset_of!(GrassBendParams, write_half), 32);
        assert_eq!(offset_of!(GrassBendParams, prev_valid), 36);
        assert_eq!(offset_of!(GrassBendParams, _pad), 40);
        assert_eq!(offset_of!(GrassBendParams, stamps), 48);
        assert!(size_of::<GrassBendParams>() <= 4096);
    }

    #[test]
    fn every_args_record_starts_as_an_empty_draw_of_its_level() {
        let args = initial_grass_args();
        let slot = [15, 0, 0, 0, 9, 0, 16, 0, 5, 0, 32, 0];
        assert_eq!(args[..12], slot);
        assert_eq!(args[12..], slot);
        assert_eq!(core::mem::size_of_val(&args), GRASS_ARGS_BYTES);
    }

    #[test]
    fn the_args_offset_alternates_between_the_two_slots() {
        assert_eq!(grass_args_offset(0), 0);
        assert_eq!(grass_args_offset(1), 48);
        assert_eq!(grass_args_offset(2), 0);
    }
}

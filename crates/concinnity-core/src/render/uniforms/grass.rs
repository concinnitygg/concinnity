//! The grass pass's blocks: the per-frame parameters every grass stage reads,
//! one visible blade as the kernel appends it, and the draw arguments the
//! kernel fills. Match `GrassParams` / `GrassBlade` in `shaders/grass.hlsl`.

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

/// Per-frame grass parameters. 256 bytes: every member a `float4` row or a
/// run of four scalars, so the constant-buffer and storage layouts agree on
/// every target.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GrassParams {
    /// The field's ground rectangle: min x, min z, max x, max z.
    pub patch_rect: [f32; 4],
    /// Height of the ground the blades root at.
    pub ground_y: f32,
    /// Edge of one tile, in meters.
    pub tile_size: f32,
    /// Edge of one cell, in meters.
    pub cell_size: f32,
    /// Cells along each tile edge.
    pub cells_per_side: u32,
    /// Tile coordinates of the first tile the dispatch covers.
    pub tile_origin: [i32; 2],
    /// Tiles the dispatch covers along x and z.
    pub tile_count: [u32; 2],
    /// Camera position the kernel measures distance from.
    pub cam_pos: [f32; 3],
    /// Distance past which no blade is kept.
    pub draw_distance: f32,
    /// The camera's world-space frustum planes, `(normal, d)` with
    /// `dot(normal, p) + d >= 0` inside.
    pub frustum: [[f32; 4]; 6],
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
    /// Blades the visible-blade buffer holds.
    pub capacity: u32,
    /// The draw-argument slot this frame fills; the other one is reset.
    pub args_slot: u32,
    /// Linear RGB at the root; `w` unused.
    pub root_color: [f32; 4],
    /// Linear RGB at the tip; `w` unused.
    pub tip_color: [f32; 4],
    /// Wind direction `[x, z]`, speed and gustiness (see `WindField::gpu_rows`).
    pub wind: [f32; 4],
    /// Inverse gust width; `yzw` unused.
    pub wind_gust: [f32; 4],
}

/// One visible blade, as the kernel appends it and the draws read it.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GpuGrassBlade {
    /// Root position (xyz) and facing angle in radians (w).
    pub root_facing: [f32; 4],
    /// Height, width at the root, static lean in [0, 1], and the blade's
    /// random bits (as a float's bit pattern).
    pub shape: [f32; 4],
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
        assert_eq!(size_of::<GrassParams>(), 256);
        assert_eq!(offset_of!(GrassParams, patch_rect), 0);
        assert_eq!(offset_of!(GrassParams, ground_y), 16);
        assert_eq!(offset_of!(GrassParams, tile_size), 20);
        assert_eq!(offset_of!(GrassParams, cell_size), 24);
        assert_eq!(offset_of!(GrassParams, cells_per_side), 28);
        assert_eq!(offset_of!(GrassParams, tile_origin), 32);
        assert_eq!(offset_of!(GrassParams, tile_count), 40);
        assert_eq!(offset_of!(GrassParams, cam_pos), 48);
        assert_eq!(offset_of!(GrassParams, draw_distance), 60);
        assert_eq!(offset_of!(GrassParams, frustum), 64);
        assert_eq!(offset_of!(GrassParams, height), 160);
        assert_eq!(offset_of!(GrassParams, height_variance), 164);
        assert_eq!(offset_of!(GrassParams, width), 168);
        assert_eq!(offset_of!(GrassParams, clump_size), 172);
        assert_eq!(offset_of!(GrassParams, stiffness), 176);
        assert_eq!(offset_of!(GrassParams, color_variation), 180);
        assert_eq!(offset_of!(GrassParams, capacity), 184);
        assert_eq!(offset_of!(GrassParams, args_slot), 188);
        assert_eq!(offset_of!(GrassParams, root_color), 192);
        assert_eq!(offset_of!(GrassParams, tip_color), 208);
        assert_eq!(offset_of!(GrassParams, wind), 224);
        assert_eq!(offset_of!(GrassParams, wind_gust), 240);
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

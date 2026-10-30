//! The clouds' constant block, as the march kernel and the composite read it.

use super::frame::{CloudFrame, WEATHER_SECOND_SCALE};
use super::settings::{CloudSettings, MAX_CLOUD_LAYERS};
use super::{CLOUD_TARGET_DIVISOR, cloud_target_size};
use crate::render::uniforms::DaylightBlock;
use concinnity_core::math::narrow_f32;

/// One layer's lanes. Mirrors `CloudLayerBlock` in `cloud_types.hlsl`.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CloudLayerBlock {
    /// x, y: the radii of its bottom and top, kilometers from the planet's
    /// center; z: its coverage; w: its towering.
    pub shell: [f32; 4],
    /// x: extinction per kilometer at full density; y, z, w: noise tiles per
    /// kilometer for the weather, the base shape and the detail.
    pub scales: [f32; 4],
    /// xyz: the weather's origin in tiles; w: the second field's tiles per
    /// kilometer.
    pub weather: [f32; 4],
    /// xyz: the weather's second field's origin in tiles.
    pub weather_second: [f32; 4],
    /// xyz: the base shape's origin in tiles.
    pub shape: [f32; 4],
    /// xyz: the detail's origin in tiles.
    pub detail: [f32; 4],
    /// xyz: the axis the clouds are drawn out along, in the planet's fixed
    /// axes; w: one over the stretch, less one.
    pub stretch: [f32; 4],
}

/// How finely the clouds are marched.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CloudMarch {
    /// Most steps along a view ray through the clouds.
    pub steps: u32,
    /// Steps along each sample's path toward a light.
    pub light_steps: u32,
    /// Farthest a view ray is marched past where it enters the clouds, in
    /// kilometers.
    pub max_distance_km: u32,
    /// Whether the clouds' edges are worn by the detail noise.
    pub detail: bool,
}

impl CloudMarch {
    /// The finest march, which the top tiers run.
    pub const FULL: Self = Self {
        steps: 96,
        light_steps: 6,
        max_distance_km: 120,
        detail: true,
    };
}

/// The clouds' constant block. Mirrors `CloudParams` in `cloud_types.hlsl`;
/// every member is a float4 lane so the three targets lay it out alike.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct CloudParams {
    /// The layers; the first `shell.z` are drawn.
    pub layers: [CloudLayerBlock; MAX_CLOUD_LAYERS],
    /// Rows of the rotation into the planet's fixed axes; w unused.
    pub to_fixed: [[f32; 4]; 3],
    /// xyz: the camera relative to the planet's center, kilometers, in its
    /// fixed axes; w: the ground's radius.
    pub camera: [f32; 4],
    /// x, y: the radii every layer lies between; z: the layer count; w: the
    /// frame's index, which turns the march's noise each frame.
    pub shell: [f32; 4],
    /// x: most view steps; y: light steps; z: farthest march, kilometers;
    /// w: 1 when the detail wears the edges.
    pub march: [f32; 4],
    /// xy: the cloud target's size in texels; zw: its reciprocal.
    pub target: [f32; 4],
    /// xy: the scene's size in pixels; zw: the pixel inside each block of
    /// [`CLOUD_TARGET_DIVISOR`] squared pixels this frame's march reads.
    pub scene: [f32; 4],
    /// xyz: the shadow map's center (fixed axes, kilometers from the planet's
    /// center); w: its width in kilometers.
    pub shadow_center: [f32; 4],
    /// xyz: the shadow map's east; w: the radius of the sphere it lies on.
    pub shadow_east: [f32; 4],
    /// xyz: the shadow map's north.
    pub shadow_north: [f32; 4],
    /// x: the radius, in kilometers, of the ground the clouds' horizon is
    /// measured from; yzw unused.
    pub horizon: [f32; 4],
    /// The light the clouds are lit by, in the planet's fixed axes.
    pub daylight: DaylightBlock,
}

/// The pixel inside each target texel's block of the scene that frame
/// `frame` marches through, so four frames cover the block.
pub fn block_pixel(frame: u64) -> [u32; 2] {
    const ORDER: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];
    let i = (frame % 4) as usize;
    ORDER[i]
}

impl CloudParams {
    /// The block for `settings` over a ground `ground_km` from the planet's
    /// center, this frame's `frame` inputs, for a `scene` pixels across and
    /// down on frame `index`.
    pub fn new(
        settings: &CloudSettings,
        ground_km: f32,
        frame: &CloudFrame,
        scene: [u32; 2],
        index: u64,
    ) -> Self {
        let march = &settings.march;
        let v3 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        let mut layers = [CloudLayerBlock::default(); MAX_CLOUD_LAYERS];
        for ((block, layer), origins) in layers.iter_mut().zip(settings.layers()).zip(&frame.layers)
        {
            *block = CloudLayerBlock {
                shell: [
                    ground_km + layer.base_km,
                    ground_km + layer.top_km,
                    layer.coverage,
                    layer.towering,
                ],
                scales: [
                    layer.extinction,
                    1.0 / layer.weather_tile_km,
                    1.0 / layer.shape_tile_km,
                    1.0 / layer.detail_tile_km,
                ],
                weather: v3(
                    origins.weather,
                    1.0 / (layer.weather_tile_km * narrow_f32(WEATHER_SECOND_SCALE)),
                ),
                weather_second: v3(origins.weather_second, 0.0),
                shape: v3(origins.shape, 0.0),
                detail: v3(origins.detail, 0.0),
                stretch: [
                    layer.stretch_axis[0],
                    0.0,
                    layer.stretch_axis[1],
                    1.0 / layer.stretch - 1.0,
                ],
            };
        }
        let [lo, hi] = settings.span_km();
        let [tw, th] = cloud_target_size(scene[0], scene[1]);
        let [px, py] = block_pixel(index);
        let [cx, cy, cz] = frame.camera_km;
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let index_lane = (index % 4096) as f32;
        Self {
            layers,
            to_fixed: frame.to_fixed.map(row),
            camera: [cx, cy, cz, ground_km],
            shell: [
                ground_km + lo,
                ground_km + hi,
                settings.count() as f32,
                index_lane,
            ],
            march: [
                march.steps as f32,
                march.light_steps as f32,
                march.max_distance_km as f32,
                if march.detail { 1.0 } else { 0.0 },
            ],
            target: [tw as f32, th as f32, 1.0 / tw as f32, 1.0 / th as f32],
            scene: [
                scene[0].max(1) as f32,
                scene[1].max(1) as f32,
                px as f32,
                py as f32,
            ],
            shadow_center: v3(frame.shadow.center_km, frame.shadow.width_km),
            shadow_east: v3(frame.shadow.east, frame.shadow.radius_km),
            shadow_north: v3(frame.shadow.north, 0.0),
            horizon: [frame.horizon_km, 0.0, 0.0, 0.0],
            daylight: frame.daylight,
        }
    }
}

const _: () = assert!(CLOUD_TARGET_DIVISOR == 2, "block_pixel walks a 2x2 block");

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::mem::{offset_of, size_of};

    #[test]
    fn cloud_layer_block_layout_matches_the_shader() {
        assert_eq!(size_of::<CloudLayerBlock>(), 112);
        assert_eq!(offset_of!(CloudLayerBlock, shell), 0);
        assert_eq!(offset_of!(CloudLayerBlock, scales), 16);
        assert_eq!(offset_of!(CloudLayerBlock, weather), 32);
        assert_eq!(offset_of!(CloudLayerBlock, weather_second), 48);
        assert_eq!(offset_of!(CloudLayerBlock, shape), 64);
        assert_eq!(offset_of!(CloudLayerBlock, detail), 80);
        assert_eq!(offset_of!(CloudLayerBlock, stretch), 96);
    }

    #[test]
    fn cloud_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CloudParams>(), 2080);
        assert_eq!(offset_of!(CloudParams, layers), 0);
        assert_eq!(offset_of!(CloudParams, to_fixed), 224);
        assert_eq!(offset_of!(CloudParams, camera), 272);
        assert_eq!(offset_of!(CloudParams, shell), 288);
        assert_eq!(offset_of!(CloudParams, march), 304);
        assert_eq!(offset_of!(CloudParams, target), 320);
        assert_eq!(offset_of!(CloudParams, scene), 336);
        assert_eq!(offset_of!(CloudParams, shadow_center), 352);
        assert_eq!(offset_of!(CloudParams, shadow_east), 368);
        assert_eq!(offset_of!(CloudParams, shadow_north), 384);
        assert_eq!(offset_of!(CloudParams, horizon), 400);
        assert_eq!(offset_of!(CloudParams, daylight), 416);
    }

    #[test]
    fn four_frames_cover_each_block() {
        let mut seen: Vec<[u32; 2]> = (0..4).map(block_pixel).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4);
        assert_eq!(block_pixel(5), block_pixel(1));
    }
}

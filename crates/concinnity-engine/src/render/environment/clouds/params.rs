//! The clouds' constant block, as the march kernel and the composite read it.

use super::frame::{CloudFrame, WEATHER_REGION_SCALE, WEATHER_SECOND_SCALE};
use super::history::CloudProjection;
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
    /// x, z: the axis the clouds are drawn out along, in the planet's fixed
    /// axes; y: their clustering; w: one over the stretch, less one.
    pub stretch: [f32; 4],
    /// xyz: the regions' origin in tiles; w: their tiles per kilometer.
    pub region: [f32; 4],
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
    /// xy: the scene's size in pixels; zw: the target texel inside each 2x2
    /// block of them this frame marches (see [`march_texel`]).
    pub scene: [f32; 4],
    /// xyz: the shadow map's center (fixed axes, kilometers from the planet's
    /// center); w: its width in kilometers.
    pub shadow_center: [f32; 4],
    /// xyz: the shadow map's east; w: the radius of the sphere it lies on.
    pub shadow_east: [f32; 4],
    /// xyz: the shadow map's north.
    pub shadow_north: [f32; 4],
    /// x: the radius, in kilometers, of the ground the clouds' horizon is
    /// measured from; y: its [`CloudHorizon::height_scale`](super::CloudHorizon::height_scale); zw unused.
    pub horizon: [f32; 4],
    /// The previous frame's [`CloudProjection::view`], which the texels this
    /// frame does not march are reprojected through.
    pub prev_view: [[f32; 4]; 4],
    /// xyz: the previous frame's camera, as `camera`; w: 1 when the previous
    /// frame's target holds its clouds.
    pub prev_camera: [f32; 4],
    /// x: the share of a marched panorama texel this frame's ray replaces
    /// (see [`Self::with_panorama_share`]); yzw unused.
    pub panorama: [f32; 4],
    /// The light the clouds are lit by, in the planet's fixed axes.
    pub daylight: DaylightBlock,
}

/// The target texel inside each 2x2 block of them that frame `frame`
/// marches, so four frames march every texel once; the rest are reprojected
/// from the previous frame's target.
pub fn march_texel(frame: u64) -> [u32; 2] {
    const ORDER: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];
    let i = (frame % 4) as usize;
    ORDER[i]
}

/// The size of the grid a frame marches over a target `target` texels across
/// and down: one texel of each 2x2 block.
pub fn cloud_march_size(target: [u32; 2]) -> [u32; 2] {
    target.map(|t| t.div_ceil(2).max(1))
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
                    layer.clustering,
                    layer.stretch_axis[1],
                    1.0 / layer.stretch - 1.0,
                ],
                region: v3(
                    origins.region,
                    1.0 / (layer.weather_tile_km * narrow_f32(WEATHER_REGION_SCALE)),
                ),
            };
        }
        let [lo, hi] = settings.span_km();
        let [tw, th] = cloud_target_size(scene[0], scene[1]);
        let [px, py] = march_texel(index);
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
            horizon: [
                frame.horizon.ground_km,
                frame.horizon.height_scale,
                0.0,
                0.0,
            ],
            prev_view: crate::transform::IDENTITY,
            prev_camera: [0.0; 4],
            panorama: [1.0, 0.0, 0.0, 0.0],
            daylight: frame.daylight,
        }
    }

    /// This block reprojecting through the previous frame's `previous`, or
    /// marching only when there is none.
    #[must_use]
    pub fn with_history(self, previous: Option<CloudProjection>) -> Self {
        match previous {
            Some(p) => {
                let [x, y, z] = p.camera_km;
                Self {
                    prev_view: p.view,
                    prev_camera: [x, y, z, 1.0],
                    ..self
                }
            }
            None => Self {
                prev_camera: [0.0; 4],
                ..self
            },
        }
    }

    /// This block with the panorama marching one texel of each 2x2 block,
    /// the one [`march_texel`] names, and replacing `share` of each with this
    /// frame's ray; a `share` of 1 marches every texel afresh.
    #[must_use]
    pub fn with_panorama_share(self, share: f32) -> Self {
        Self {
            panorama: [share, 0.0, 0.0, 0.0],
            ..self
        }
    }
}

const _: () = assert!(
    CLOUD_TARGET_DIVISOR == 2,
    "the composite reads each target texel's top-left pixel"
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::mem::{offset_of, size_of};

    #[test]
    fn cloud_layer_block_layout_matches_the_shader() {
        assert_eq!(size_of::<CloudLayerBlock>(), 128);
        assert_eq!(offset_of!(CloudLayerBlock, shell), 0);
        assert_eq!(offset_of!(CloudLayerBlock, scales), 16);
        assert_eq!(offset_of!(CloudLayerBlock, weather), 32);
        assert_eq!(offset_of!(CloudLayerBlock, weather_second), 48);
        assert_eq!(offset_of!(CloudLayerBlock, shape), 64);
        assert_eq!(offset_of!(CloudLayerBlock, detail), 80);
        assert_eq!(offset_of!(CloudLayerBlock, stretch), 96);
        assert_eq!(offset_of!(CloudLayerBlock, region), 112);
    }

    #[test]
    fn cloud_params_layout_matches_the_shader() {
        assert_eq!(size_of::<CloudParams>(), 2208);
        assert_eq!(offset_of!(CloudParams, layers), 0);
        assert_eq!(offset_of!(CloudParams, to_fixed), 256);
        assert_eq!(offset_of!(CloudParams, camera), 304);
        assert_eq!(offset_of!(CloudParams, shell), 320);
        assert_eq!(offset_of!(CloudParams, march), 336);
        assert_eq!(offset_of!(CloudParams, target), 352);
        assert_eq!(offset_of!(CloudParams, scene), 368);
        assert_eq!(offset_of!(CloudParams, shadow_center), 384);
        assert_eq!(offset_of!(CloudParams, shadow_east), 400);
        assert_eq!(offset_of!(CloudParams, shadow_north), 416);
        assert_eq!(offset_of!(CloudParams, horizon), 432);
        assert_eq!(offset_of!(CloudParams, prev_view), 448);
        assert_eq!(offset_of!(CloudParams, prev_camera), 512);
        assert_eq!(offset_of!(CloudParams, panorama), 528);
        assert_eq!(offset_of!(CloudParams, daylight), 544);
    }

    #[test]
    fn four_frames_march_each_texel_of_a_block() {
        let mut seen: Vec<[u32; 2]> = (0..4).map(march_texel).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4);
        assert_eq!(march_texel(5), march_texel(1));
    }

    #[test]
    fn the_march_covers_one_texel_of_every_block() {
        assert_eq!(cloud_march_size([960, 540]), [480, 270]);
        assert_eq!(cloud_march_size([7, 1]), [4, 1]);
        assert_eq!(cloud_march_size([0, 0]), [1, 1]);
    }

    // One default layer in a still wind, the march at full quality.
    fn one_layer() -> CloudSettings {
        CloudSettings::resolve(
            &[crate::components::Cloud::default()],
            &crate::wind::WindField::STILL,
            CloudMarch::FULL,
        )
        .expect("a default cloud resolves")
    }

    fn frame(camera_km: [f32; 3]) -> CloudFrame {
        CloudFrame {
            layers: Default::default(),
            to_fixed: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            camera_km,
            daylight: DaylightBlock::OFF,
            shadow: super::super::shadow::CloudShadowSquare::default(),
            shadow_lanes: crate::gfx::render_types::CloudShadowLanes::OFF,
            time: 0.0,
            horizon: super::super::frame::CloudHorizon::true_ground(6360.0),
        }
    }

    // A layer's shell is its base and top above the ground, and the shell
    // around every layer spans the lowest base to the highest top.
    #[test]
    fn layers_sit_above_the_ground_they_are_built_over() {
        let settings = one_layer();
        let layer = settings.layers()[0];
        let ground = 6360.0;
        let p = CloudParams::new(
            &settings,
            ground,
            &frame([0.0, ground, 0.0]),
            [1280, 720],
            0,
        );
        assert_eq!(p.layers[0].shell[0], ground + layer.base_km);
        assert_eq!(p.layers[0].shell[1], ground + layer.top_km);
        let [lo, hi] = settings.span_km();
        assert_eq!(&p.shell[..3], &[ground + lo, ground + hi, 1.0]);
        assert_eq!(p.camera, [0.0, ground, 0.0, ground]);
        assert_eq!(p.layers[1].shell, [0.0; 4], "an undeclared layer is empty");
    }

    // The target is the scene scaled down, with its reciprocals beside it, and
    // the frame index picks the texel each 2x2 block marches.
    #[test]
    fn the_target_and_the_marched_texel_follow_the_scene_and_frame() {
        let settings = one_layer();
        let p = CloudParams::new(&settings, 6360.0, &frame([0.0; 3]), [1280, 720], 2);
        assert_eq!(p.target, [640.0, 360.0, 1.0 / 640.0, 1.0 / 360.0]);
        assert_eq!(
            p.scene,
            [1280.0, 720.0, 1.0, 0.0],
            "frame 2 marches texel [1, 0]"
        );
        assert_eq!(p.shell[3], 2.0, "the frame index lane");
        let wrapped = CloudParams::new(&settings, 6360.0, &frame([0.0; 3]), [1280, 720], 4098);
        assert_eq!(wrapped.shell[3], 2.0, "the index lane wraps at 4096");
    }

    // With no previous frame the block marches; with one it reprojects
    // through the previous view from the previous camera.
    #[test]
    fn history_is_read_only_when_there_is_a_previous_frame() {
        let settings = one_layer();
        let p = CloudParams::new(&settings, 6360.0, &frame([0.0; 3]), [64, 64], 0);
        let mut moved = crate::transform::IDENTITY;
        moved[3] = [4.0, 5.0, 6.0, 1.0];
        let previous = CloudProjection {
            view: moved,
            camera_km: [1.0, 2.0, 3.0],
        };
        let reprojecting = p.with_history(Some(previous));
        assert_eq!(reprojecting.prev_view, moved);
        assert_eq!(reprojecting.prev_camera, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            reprojecting.with_history(None).prev_camera,
            [0.0; 4],
            "losing the previous frame turns reprojection off"
        );
        assert_eq!(
            p.panorama[0], 1.0,
            "every panorama texel marches by default"
        );
        assert_eq!(p.with_panorama_share(0.25).panorama[0], 0.25);
    }
}

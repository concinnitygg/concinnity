//! Shadow layouts: cascade, far and spot shadow uniforms and the shadow array
//! extent.

/// Number of cascades the directional shadow pre-pass renders into the shadow
/// map array. Hardcoded because changing N requires re-compiling the shaders
/// (the array length appears in the MSL/HLSL/GLSL source).
pub const NUM_SHADOW_CASCADES: usize = 4;

/// Slices in the spot shadow map array: the number of spot lights that can cast
/// shadows at once. Spots past this still light the scene with `shadow_index`
/// left at -1. Hardcoded for the same reason as `NUM_SHADOW_CASCADES`.
pub const MAX_SHADOWED_SPOTS: usize = 16;

/// Per-slice edge of the spot shadow map array, derived from the authored
/// directional `shadow_map_size` so one quality knob scales both. Quartered
/// because a spot's cone covers far less world area than a CSM cascade, and the
/// array multiplies the cost by `MAX_SHADOWED_SPOTS`.
pub fn spot_shadow_slice_size(shadow_map_size: u32) -> u32 {
    (shadow_map_size / 4).clamp(256, 1024)
}

/// The allocated shape of a shadow map array: texel edge and slice count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowArrayExtent {
    /// Texel edge of every slice.
    pub size: u32,
    /// Number of slices.
    pub layers: u32,
}

/// The array allocated for `layers` slices of `size` texels. An empty array
/// (size or layer count 0, i.e. shadows off) becomes a single 1x1 slice, so
/// every view over it names layers that exist.
pub fn shadow_array_extent(size: u32, layers: u32) -> ShadowArrayExtent {
    if size == 0 || layers == 0 {
        ShadowArrayExtent { size: 1, layers: 1 }
    } else {
        ShadowArrayExtent { size, layers }
    }
}

/// Cascaded shadow map view-projection matrices and split depths.
///
/// Pushed to the shadow-pass vertex shader at buffer(0) (alongside a
/// `cascade_idx` push constant that picks one matrix) and to the main-pass
/// fragment shader at buffer(5). Layout must stay in sync with the
/// `ShadowUniforms` struct in `light_types.hlsl`, which every shader that binds
/// it shares.
///
/// `cascade_splits` are the view-space depth thresholds (positive) marking the
/// FAR end of each cascade. The fragment shader picks the first cascade whose
/// split is greater than the fragment's view-space depth. The last live
/// cascade's far end is the overall shadow reach.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ShadowUniforms {
    /// One light-space VP matrix per cascade, column-major.
    pub light_vps: [[[f32; 4]; 4]; NUM_SHADOW_CASCADES],
    /// View-space depth at the FAR end of each cascade. Stored as a vec4 to
    /// keep MSL std140-like alignment trivial across backends. Slots
    /// `[active_cascades..]` hold a negative sentinel so the fragment shader's
    /// split comparison never selects an unrendered cascade.
    pub cascade_splits: [f32; NUM_SHADOW_CASCADES],
    /// How many of the `NUM_SHADOW_CASCADES` slots are live this frame (1..=4,
    /// from `Shadows.cascades`). The array capacity stays 4; only
    /// the first `active_cascades` are split, rendered, and sampled. The fragment
    /// shader bounds both its cascade fallback and its cross-cascade blend by this
    /// so it never reads a slot the CPU did not render.
    pub active_cascades: u32,
    /// View-space depth past which the cascaded shadow fades out, reaching
    /// none at the last live split. Infinite when nothing fades.
    pub fade_start: f32,
    /// Fills the scalars' lane from 280 to 288, so `clouds` starts on the
    /// 16-byte boundary the shaders place it at.
    pub _pad: [u32; 2],
    /// Where the clouds' shadow map lies over the ground around the camera.
    pub clouds: CloudShadowLanes,
    /// The far shadow past the cascades.
    pub far: FarShadowLanes,
}

/// The far shadow map as the shaded passes read it: one orthographic view of
/// the large casters around where the camera stood when it was drawn, sampled
/// where the cascades fade out. 96 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct FarShadowLanes {
    /// World to the map's clip space, column-major.
    pub light_vp: [[f32; 4]; 4],
    /// xyz: the center its coverage is measured from; w: the distance from it
    /// where the far shadow is gone, 0 when there is none.
    pub center: [f32; 4],
    /// The distance from `center` where the far shadow starts fading out.
    pub fade_start: f32,
    /// The depth compare offset, in the map's depth units.
    pub compare_offset: f32,
    /// One texel, in map coordinates.
    pub texel: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

impl FarShadowLanes {
    /// No far shadow: past the cascades every point is fully lit.
    pub const OFF: Self = Self {
        light_vp: [[0.0; 4]; 4],
        center: [0.0; 4],
        fade_start: 0.0,
        compare_offset: 0.0,
        texel: 0.0,
        _pad: 0.0,
    };
}

/// The clouds' shadow map as the shaded passes read it, in the frame's axes
/// and meters: a square on the sphere through the middle of the cloud layers,
/// reached from a point along the shadowed light. 64 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct CloudShadowLanes {
    /// xyz: the square's center; w: the radius of the sphere it lies on.
    pub center: [f32; 4],
    /// xyz: the square's east; w: one over its width.
    pub east: [f32; 4],
    /// xyz: the square's north; w: 1 when the clouds cast shadows.
    pub north: [f32; 4],
    /// xyz: toward the shadowed light; w unused.
    pub light: [f32; 4],
}

impl CloudShadowLanes {
    /// No cloud shadows: the shaded passes skip the map.
    pub const OFF: Self = Self {
        center: [0.0; 4],
        east: [0.0; 4],
        north: [0.0; 4],
        light: [0.0; 4],
    };
}

/// Per-view constants of the scene shadow pass, pushed once per cascade or
/// spot slice on every backend: which of `ShadowUniforms.light_vps` the view
/// projects through, and the camera a planet tile casts as the main pass draws
/// it. 48 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ShadowPassPush {
    /// Which cascade the shadow pass is rendering.
    pub cascade_idx: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 3],
    /// The camera a planet's tiles morph by.
    pub tile_morph: super::TileMorphView,
}

/// One shadowed spot light's slice of the spot shadow map array. Indexed by
/// `GpuLight.shadow_index`; the index doubles as the array slice. Uploaded once
/// per scene, since local lights are static.
///
/// 80 bytes: the matrix fills the first four 16-byte lanes and the three scalars
/// plus a pad fill the fifth, so the MSL (float4x4 + floats), HLSL (no 16-byte
/// straddle), and GLSL std430 (mat4 align-16) layouts all agree.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct SpotShadowData {
    /// World -> light clip matrix for this spot, column-major.
    pub light_vp: [[f32; 4]; 4],
    /// Constant depth offset applied when comparing against the stored depth.
    pub depth_bias: f32,
    /// Offset along the surface normal before projecting, which suppresses
    /// shadow acne on surfaces near-parallel to the light.
    pub normal_bias: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [f32; 2],
}

impl SpotShadowData {
    /// An all-zero slice, used to fill unused entries.
    pub const ZERO: Self = Self {
        light_vp: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        depth_bias: 0.0,
        normal_bias: 0.0,
        _pad: [0.0; 2],
    };
}

/// Texels across the square of the depth one group of the shadow receiver
/// reduction covers.
pub const RECEIVER_TILE: u32 = 64;

/// The fewest lanes a wave may have for the shadow receiver reduction to run:
/// its group memory holds one entry per wave down to this width.
pub const RECEIVER_MIN_LANES: u32 = 4;

/// The bound a receiver band no surface fell in keeps at both ends: its low
/// end above its high one.
pub const RECEIVER_EMPTY: f32 = 3.0e38;

/// The tiles the shadow receiver reduction splits a `width` by `height` depth
/// into, across and down: one group and one [`ReceiverBands`] each.
pub fn receiver_tiles(width: u32, height: u32) -> [u32; 2] {
    [
        width.div_ceil(RECEIVER_TILE),
        height.div_ceil(RECEIVER_TILE),
    ]
}

/// What the shadow receiver reduction reads: how a depth texel becomes a
/// light-space point and which cascade depth bands it falls in. 128 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct ReceiverBoundsParams {
    /// NDC x, y and device depth to the light's right, up and toward-the-light
    /// coordinates and the view depth, each times the point's homogeneous w;
    /// column-major.
    pub to_light_space: [[f32; 4]; 4],
    /// NDC x, y and device depth to the point's homogeneous w.
    pub w_row: [f32; 4],
    /// View depth where each band starts.
    pub band_near: [f32; 4],
    /// View depth where each band ends; before it starts for a band not in use.
    pub band_far: [f32; 4],
    /// The depth's width in texels.
    pub width: u32,
    /// The depth's height in texels.
    pub height: u32,
    /// Tiles across the depth ([`receiver_tiles`]).
    pub groups_x: u32,
    /// The measurement's number, written back with its bounds.
    pub serial: u32,
}

/// The surfaces one cascade's depth band holds: the least and greatest of
/// their right, up and toward-the-light coordinates and their view depth.
/// [`RECEIVER_EMPTY`] at both ends for none. 32 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ReceiverBand {
    /// The least of each coordinate.
    pub lo: [f32; 4],
    /// The greatest of each coordinate.
    pub hi: [f32; 4],
}

impl ReceiverBand {
    /// A band no surface fell in.
    pub const EMPTY: Self = Self {
        lo: [RECEIVER_EMPTY; 4],
        hi: [-RECEIVER_EMPTY; 4],
    };

    /// Whether no surface fell in the band.
    pub fn is_empty(&self) -> bool {
        (0..4).any(|i| self.lo[i] > self.hi[i])
    }
}

/// One tile's bounds in every band, as the reduction writes them for the
/// merge. 128 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ReceiverBands {
    /// Each cascade's band.
    pub bands: [ReceiverBand; NUM_SHADOW_CASCADES],
}

/// The surfaces the camera saw in each cascade's depth band, as the merge
/// writes them for the CPU to read back. 144 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ReceiverBounds {
    /// Each cascade's band.
    pub bands: [ReceiverBand; NUM_SHADOW_CASCADES],
    /// The measurement's number, from its [`ReceiverBoundsParams`].
    pub serial: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 3],
}

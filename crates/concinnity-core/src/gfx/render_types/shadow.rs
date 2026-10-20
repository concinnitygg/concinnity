//! Shadow layouts: cascade and spot shadow uniforms and the shadow array extent.

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
/// `ShadowUniforms` struct in the single-source shaders. Three of them declare
/// it: the forward pass stops at `active_cascades`, the shadow pass at
/// `cascade_splits`, and only the fog froxel kernel spells out the trailing pad.
///
/// `cascade_splits` are the view-space depth thresholds (positive) marking the
/// FAR end of each cascade. The fragment shader picks the first cascade whose
/// split is greater than the fragment's view-space depth. Slot 3 (the last
/// cascade's far end) doubles as the overall shadow distance.
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
    /// Pads the struct to 288 bytes (a multiple of 16). The Rust `[[f32; 4]; 4]`
    /// matrices are only 4-byte aligned, so Rust would otherwise leave this at 276;
    /// the MSL `float4x4`-aligned struct and the GLSL std140 block both round up to
    /// 288, so the upload size must match for the bound buffer range to cover them.
    pub _pad: [u32; 3],
}

/// Per-shadow-pass push constant identifying which cascade is being rendered.
/// Used so the shadow vertex shader can index `ShadowUniforms.light_vps[i]`
/// from a single bound UBO instead of re-binding a different uniform per pass.
///
/// Consumed at runtime only by the Metal shadow pass (Vulkan and DirectX each
/// define their own private push-constant layouts in their respective `draw.rs`
/// modules), but kept un-gated like the rest of this module so the shared
/// shader-layout contract can validate its MSL layout on every platform.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ShadowPassPush {
    /// Which cascade the shadow pass is rendering.
    pub cascade_idx: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 3],
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

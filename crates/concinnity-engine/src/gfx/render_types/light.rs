//! Light layouts: the light block, the clustered light grid, and area lights.

use concinnity_core::math::SaturatingFrom;

/// Capacity of the fixed directional-light array in `LightUniforms`.
pub const MAX_DIRECTIONAL_LIGHTS: usize = 4;
/// Capacity of the fixed point-light array in `LightUniforms`, which world
/// Shader hooks read.
pub const MAX_POINT_LIGHTS: usize = 8;

/// Capacity of the per-scene local-light storage buffer the forward pass reads.
/// Distinct from `MAX_POINT_LIGHTS`, which bounds the fixed `LightUniforms` point
/// array world Shader hooks read. Lights past this
/// cap are dropped with a warning.
pub const MAX_LOCAL_LIGHTS: usize = 1024;

/// GpuLight.kind discriminants.
pub const LIGHT_KIND_POINT: u32 = 0;
/// `GpuLight.kind` for a spot light. See [`LIGHT_KIND_POINT`].
pub const LIGHT_KIND_SPOT: u32 = 1;
/// `GpuLight.kind` for a rectangular area light. See [`LIGHT_KIND_POINT`].
pub const LIGHT_KIND_AREA: u32 = 2;

/// Capacity of the per-scene `AreaLightData` table. Area lights are far heavier to
/// shade than point or spot lights (a polygon integral per fragment), so the cap
/// is deliberately well under `MAX_LOCAL_LIGHTS`.
pub const MAX_AREA_LIGHTS: usize = 64;

/// Clustered forward lighting froxel grid. The screen is tiled
/// `CLUSTER_GRID_X` x `CLUSTER_GRID_Y` with `CLUSTER_GRID_Z` exponential depth slices;
/// a compute pass bins the local lights into per-cluster index lists the forward
/// pass reads instead of iterating every light.
pub(super) const CLUSTER_GRID_X: u32 = 16;
/// Cluster tiles down the screen. See [`CLUSTER_GRID_X`].
pub(super) const CLUSTER_GRID_Y: u32 = 9;
/// Exponential depth slices per cluster column. See [`CLUSTER_GRID_X`].
pub(super) const CLUSTER_GRID_Z: u32 = 24;
/// Total clusters in the froxel grid.
pub const CLUSTER_COUNT: u32 = CLUSTER_GRID_X * CLUSTER_GRID_Y * CLUSTER_GRID_Z;
/// Per-cluster light-index list capacity. Each cluster occupies
/// `CLUSTER_LIGHT_LIST_STRIDE` u32 slots: slot 0 is the count, slots 1.. are light
/// indices into the `GpuLight` buffer. Lights past the cap are dropped.
pub(super) const MAX_LIGHTS_PER_CLUSTER: u32 = 63;
/// `u32` slots one cluster occupies in the light-index buffer: the count plus
/// [`MAX_LIGHTS_PER_CLUSTER`] indices.
pub(super) const CLUSTER_LIGHT_LIST_STRIDE: u32 = MAX_LIGHTS_PER_CLUSTER + 1;
/// `u32` words of each of a cluster's two reflection-probe masks. The masks
/// follow every cluster's light list in the same buffer: bit `i` of a cluster's
/// influence mask is set when probe `i`'s influence reaches the cluster, and bit
/// `i` of its nearest mask when probe `i` can be the nearest capture to a point
/// in it (the fallback where no influence reaches).
pub(super) const CLUSTER_PROBE_MASK_WORDS: u32 = 8;
/// Probes a cluster mask can exclude. A probe past this index is never masked
/// out, so a larger set costs its readers time but never drops a reflection.
#[cfg(test)]
pub(super) const MAX_CLUSTERED_PROBES: u32 = CLUSTER_PROBE_MASK_WORDS * 32;
/// `u32` slots in the whole per-cluster list buffer: every cluster's light list,
/// then every cluster's two probe masks.
pub const CLUSTER_LIST_LEN: u32 =
    CLUSTER_COUNT * (CLUSTER_LIGHT_LIST_STRIDE + 2 * CLUSTER_PROBE_MASK_WORDS);

/// One directional light entry in `LightUniforms`.
/// Layout (32 bytes) mirrors `DirLight` in `main_types.hlsl`, which spells each
/// (float3, scalar) pair as one float4 lane.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DirectionalLightData {
    /// Unit vector pointing TOWARD the light source (same as L in Blinn-Phong).
    pub direction: [f32; 3],
    /// Radiance scale.
    pub intensity: f32,
    /// Linear RGB light color.
    pub color: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

/// One point light entry in `LightUniforms`.
/// Layout (32 bytes) mirrors `PointLight` in `main_types.hlsl`, spelled the
/// same way as `DirectionalLightData`.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PointLightData {
    /// World-space position of the light source.
    pub position: [f32; 3],
    /// Maximum reach in meters; attenuation is zero at this distance.
    pub range: f32,
    /// Linear RGB light color.
    pub color: [f32; 3],
    /// Radiance scale.
    pub intensity: f32,
}

/// One local light in the per-scene storage buffer the forward pass iterates
/// (bound at Metal fragment buffer(8)). 64 bytes = four 16-byte lanes, so every
/// `packed_float3` sits inside one lane with no GPU alignment promotion. Must match
/// the `GpuLight` struct in the single-source shaders.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GpuLight {
    /// World-space position (point / spot).
    pub position: [f32; 3],
    /// Maximum reach in meters; attenuation reaches zero at this distance.
    pub range: f32,
    /// Linear RGB light color.
    pub color: [f32; 3],
    /// Radiance scale.
    pub intensity: f32,
    /// Unit vector the light points along (spot / area). Zero for a point light.
    pub direction: [f32; 3],
    /// `LIGHT_KIND`_* discriminant.
    pub kind: u32,
    /// Spot cone: cosine of the inner (full-bright) and outer (cutoff) half-angles.
    /// Both zero for a point light; the shader gates the cone term on `kind`.
    pub cos_inner: f32,
    /// Cosine of the spot cone's outer (cutoff) half-angle.
    pub cos_outer: f32,
    /// Index into the spot shadow atlas, or -1 when the light casts no shadow.
    pub shadow_index: i32,
    /// Index into the `AreaLightData` table for an area light, or -1 for a point /
    /// spot light. A rect needs two edge vectors that do not fit in the spare
    /// fields here, so the extra data lives in a parallel table (the same shape
    /// `shadow_index` uses for the spot shadow slices).
    pub data_index: i32,
}

impl GpuLight {
    /// An all-zero light, used to fill unused slots.
    pub const ZERO: Self = Self {
        position: [0.0; 3],
        range: 0.0,
        color: [0.0; 3],
        intensity: 0.0,
        direction: [0.0; 3],
        kind: LIGHT_KIND_POINT,
        cos_inner: 0.0,
        cos_outer: 0.0,
        shadow_index: -1,
        data_index: -1,
    };
}

const ZERO_DIR_LIGHT: DirectionalLightData = DirectionalLightData {
    direction: [0.0; 3],
    intensity: 0.0,
    color: [0.0; 3],
    _pad: 0.0,
};

const ZERO_POINT_LIGHT: PointLightData = PointLightData {
    position: [0.0; 3],
    range: 0.0,
    color: [0.0; 3],
    intensity: 0.0,
};

/// All scene lights packed into a single GPU buffer pushed at fragment buffer(4).
/// Mirrors `LightUniforms` in `main_types.hlsl`; the layout guard reflects it.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct LightUniforms {
    /// Directional lights, the first `num_directional` entries live.
    pub directional: [DirectionalLightData; MAX_DIRECTIONAL_LIGHTS],
    /// Point lights, the first `num_point` entries live.
    pub point: [PointLightData; MAX_POINT_LIGHTS],
    /// Live entries in `directional`.
    pub num_directional: i32,
    /// Live entries in `point`.
    pub num_point: i32,
    /// Multiplier on the indirect (IBL / flat-fallback) ambient term in the main
    /// pass, resolved from `IndirectLighting.ambient_intensity`. 1.0 leaves the
    /// physically derived ambient untouched; higher values lift fill in shadowed
    /// areas the directional light cannot reach. Occupies the first of the two
    /// trailing pad words, so the 400-byte layout is unchanged.
    pub ambient_intensity: f32,
    /// Number of valid entries in the `GpuLight` storage buffer the forward pass
    /// reads. Occupies the second trailing pad word, so the 400-byte layout is
    /// unchanged. Shader hooks read `num_point` against the fixed `point` array
    /// instead.
    pub num_local_lights: i32,
}

impl LightUniforms {
    /// Neutral directional sun; used when no Light components are declared.
    pub const DEFAULT: Self = Self {
        directional: [
            DirectionalLightData {
                direction: [-0.3, 0.85, 0.4],
                intensity: 1.0,
                color: [1.0, 1.0, 1.0],
                _pad: 0.0,
            },
            ZERO_DIR_LIGHT,
            ZERO_DIR_LIGHT,
            ZERO_DIR_LIGHT,
        ],
        point: [
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
            ZERO_POINT_LIGHT,
        ],
        num_directional: 1,
        num_point: 0,
        ambient_intensity: 1.0,
        num_local_lights: 0,
    };
}

/// Per-frame uniform for the clustered light-binning compute pass and the forward
/// pass that reads the per-cluster lists. The compute kernel unprojects each
/// cluster's screen tile through `inv_view_proj` and walks camera rays to the
/// slice's near/far depth to build a world-space AABB (same convention as the fog
/// froxel kernel), then tests each `GpuLight` sphere against it. The forward pass
/// reads the grid dims + depth range + screen size to map a fragment to its
/// cluster. The same pass bins the reflection probes' influence boxes into
/// per-cluster probe masks. 128 bytes; `cam_pos` / `view_forward` each share a
/// 16-byte lane with the scalar after them, which the shader spells as one
/// float4. Must match the `ClusterParams` struct in `shaders/cluster_types.hlsl`.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct ClusterParams {
    /// Inverse view-projection; unprojects a clip-space corner to world space.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera world position; ray origin for the cluster-AABB build.
    pub cam_pos: [f32; 3],
    /// Camera near plane in view units (positive distance).
    pub z_near: f32,
    /// Camera forward axis (world space, unit); projects a ray onto view depth.
    pub view_forward: [f32; 3],
    /// Far end of the sliced depth range in view units (positive distance):
    /// the reach of what the grid bins, from
    /// `ClusterReach::range`.
    /// The last slice also holds every fragment past it.
    pub z_far: f32,
    /// Cluster tiles across the screen.
    pub grid_x: u32,
    /// Cluster tiles down the screen.
    pub grid_y: u32,
    /// Exponential depth slices per cluster column.
    pub grid_z: u32,
    /// Number of valid `GpuLight` entries the kernel bins.
    pub num_lights: u32,
    /// Render-target size in pixels; maps a fragment's `position.xy` to a tile.
    pub screen_w: f32,
    /// Render-target height in pixels.
    pub screen_h: f32,
    /// 1 = the forward pass reads the per-cluster list (main camera); 0 = it
    /// falls back to iterating all `num_lights` lights (planar / probe re-renders,
    /// whose viewpoint differs from the grid the main camera binned).
    pub use_clusters: u32,
    /// Number of reflection-probe records the kernel bins: the frame's
    /// `ProbeSet::count`.
    pub num_probes: u32,
}

impl ClusterParams {
    /// Neutral params for context construction before the first per-frame update.
    /// `use_clusters` = 0 so the forward pass falls back to brute-force iteration
    /// until a real frame's camera fills the grid.
    pub const ZERO: Self = Self {
        inv_view_proj: [[0.0; 4]; 4],
        cam_pos: [0.0; 3],
        z_near: 0.0,
        view_forward: [0.0; 3],
        z_far: 0.0,
        grid_x: CLUSTER_GRID_X,
        grid_y: CLUSTER_GRID_Y,
        grid_z: CLUSTER_GRID_Z,
        num_lights: 0,
        screen_w: 0.0,
        screen_h: 0.0,
        use_clusters: 0,
        num_probes: 0,
    };

    /// The main camera's params for this frame. `use_clusters` is set while
    /// [`clustering_active`](crate::render::light::lights::clustering_active) holds for
    /// the counts; a negative `num_local_lights` bins none.
    pub fn for_camera(camera: &ClusterCamera, num_local_lights: i32, num_probes: u32) -> Self {
        Self {
            num_lights: u32::saturating_from(num_local_lights),
            use_clusters: u32::from(crate::render::light::lights::clustering_active(
                num_local_lights,
                num_probes as usize,
            )),
            num_probes,
            ..Self::ZERO
        }
        .with_camera(camera)
    }

    /// The same lights and probes binned over `camera`'s grid instead: a
    /// second viewpoint, such as a planar mirror, shading the frame's lights.
    #[must_use]
    pub fn with_camera(&self, camera: &ClusterCamera) -> Self {
        let view = camera.view;
        Self {
            inv_view_proj: crate::transform::mat4_inverse(crate::transform::mat4_mul(
                camera.proj,
                view,
            )),
            cam_pos: camera.position,
            z_near: camera.near.max(1e-3),
            view_forward: [-view[0][2], -view[1][2], -view[2][2]],
            z_far: camera.range,
            screen_w: camera.width as f32,
            screen_h: camera.height as f32,
            ..*self
        }
    }
}

/// The camera a frame's cluster grid is binned for.
#[derive(Copy, Clone, Debug)]
pub struct ClusterCamera {
    /// World -> view, column-major.
    pub view: [[f32; 4]; 4],
    /// The un-jittered projection.
    pub proj: [[f32; 4]; 4],
    /// Camera world position.
    pub position: [f32; 3],
    /// Near plane in view units; clamped to 1e-3 so the slice math stays finite.
    pub near: f32,
    /// Far end of the sliced depth range in view units.
    pub range: f32,
    /// Render-target width in pixels.
    pub width: u32,
    /// Render-target height in pixels.
    pub height: u32,
}

/// One rectangular area light's extent, indexed by `GpuLight.data_index`. The
/// center is the `GpuLight`'s `position` and the emitting direction its
/// `direction`; only the two in-plane edge vectors and the sidedness flag need
/// the extra room.
///
/// 32 bytes: a scalar follows each vec3 so the MSL (`packed_float3`), HLSL (float3,
/// no 16-byte straddle), and GLSL std430 (vec3 align-16) layouts all agree.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct AreaLightData {
    /// Half-width times the panel's width axis, in world units.
    pub right: [f32; 3],
    /// 1 when the panel emits from both faces, 0 when it emits only along
    /// `GpuLight.direction`.
    pub two_sided: u32,
    /// Half-height times the panel's height axis, in world units.
    pub up: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

impl AreaLightData {
    /// An all-zero entry, used to fill unused slots.
    pub const ZERO: Self = Self {
        right: [0.0; 3],
        two_sided: 0,
        up: [0.0; 3],
        _pad: 0.0,
    };
}

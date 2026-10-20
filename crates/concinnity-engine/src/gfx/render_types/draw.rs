//! Draw layouts: draw objects, packed per-object records, indirect draw arguments,
//! and instanced and skinned draws.

use super::material::{FarFieldTint, MaterialUniforms, albedo_pool_index, normal_pool_index};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

/// Maximum number of joints in a single skinned-mesh skeleton. Enforced
/// CPU-side as a clamp on each `SkinnedDrawObject.joint_count` and on the
/// matching `skinned_joint_matrices` Vec length. The skinned shaders read
/// the joints buffer through a pointer (`constant float4x4 *joints`) using
/// vertex-encoded joint indices, so the GPU buffer size and joint count are
/// fully dynamic: this constant just caps how many matrices the per-frame
/// upload may carry per object.
pub const MAX_JOINTS: usize = 64;

/// Runtime-clone cap: how many spawned clones the GPU-driven cull records
/// reserve room for, past the streamed-chunk window.
///
/// Exhausting it takes 129+ live clones at once, which only runtime spawn or
/// `world.jsonl` hot-reload churn reaches; the backend's
/// `clone_static_draw_object` errors once the reserve is full (see
/// [`runtime_reserve_full`]).
pub const MAX_CLONE_DRAWS: usize = 128;

/// How many clone records a world needs reserved. A clone copies an existing
/// draw object's geometry slice, so a world with no build-time objects has
/// nothing to clone from and reserves none -- which is what keeps `cull_count()`
/// zero for a world with no geometry at all, and so keeps the GPU-driven path
/// (and the pipelines it builds) off for a menu-only world.
pub fn clone_reserve(n_objects: usize) -> usize {
    if n_objects == 0 { 0 } else { MAX_CLONE_DRAWS }
}

/// Whether the runtime reserve has no record left for another object. Streamed
/// chunks and clones share it, and a retired slot frees its record, so what
/// counts is how many objects past the build-time count are resident now.
pub fn runtime_reserve_full(objects: &[DrawObject], n_objects: usize, n_runtime: usize) -> bool {
    objects
        .iter()
        .skip(n_objects)
        .filter(|o| o.resident)
        .count()
        >= n_runtime
}

/// One LOD level past LOD0 for a `DrawObject`. Holds the rebased shared-index
/// buffer slice and the camera-distance threshold above which the renderer
/// picks this slice instead of the one stored on the parent `DrawObject`.
/// `LodSlice`s are stored in ascending `switch_distance` order; the runtime
/// picks the highest-indexed slice whose threshold is ≤ the current camera
/// distance. The same `base_vertex` as the parent applies: LOD decimation
/// reuses the LOD0 vertex range, only the index list changes.
#[derive(Copy, Clone, Debug)]
pub struct LodSlice {
    /// First index of this LOD's slice in the shared index buffer.
    pub index_offset: usize,
    /// Index count of this LOD's slice.
    pub index_count: usize,
    /// Camera distance above which the renderer picks this slice.
    pub switch_distance: f32,
}

macro_rules! draw_list_index {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            /// The position this index names in its draw-object list.
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// Wrap a draw-object list position. Debug builds assert that it
            /// fits in a `u32`.
            pub fn from_usize(i: usize) -> Self {
                debug_assert!(
                    u32::try_from(i).is_ok(),
                    concat!(stringify!($name), " does not fit in a u32")
                );
                Self(u32::saturating_from(i))
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

draw_list_index! {
    /// A slot in a backend's static draw-object list ([`DrawObject`]).
    DrawIndex
}

draw_list_index! {
    /// A slot in a backend's skinned draw-object list ([`SkinnedDrawObject`]).
    SkinnedIndex
}

/// One renderable object: vertex/index slice within the shared GPU buffers,
/// a model matrix, albedo and normal-map texture slots, and material parameters.
pub struct DrawObject {
    /// Byte offset into the shared vertex buffer.
    pub vertex_offset: usize,
    /// Number of vertices this object occupies in the shared vertex buffer,
    /// starting at `vertex_offset`. Used by mesh streaming to memcpy the
    /// object's geometry region in and out of the GPU buffer.
    pub vertex_count: usize,
    /// Element offset into the shared index buffer.
    pub index_offset: usize,
    /// Number of indices to draw.
    pub index_count: usize,
    /// The planet tile the draw is, when it is one: its vertices morph toward
    /// its parent's mesh.
    pub planet_tile: Option<PlanetTileDraw>,
    /// Value added to every index before the vertex fetch (the
    /// `drawIndexedPrimitives` `baseVertex` / D3D12 `BaseVertexLocation` /
    /// Vulkan `vertex_offset`). 0 for static and streamed-mesh geometry, whose
    /// indices are already absolute into the shared vertex buffer. Streamed
    /// `VoxelWorld` chunks instead keep mesh-relative (0-based) indices and set
    /// this to their vertex region's base, so a chunk placed past the
    /// 65 535-vertex `u16` index range still renders. Read by all three
    /// backends' shadow/main/velocity passes.
    pub base_vertex: i32,
    /// Bumped every time the slot's vertex / index bytes are rewritten in place
    /// at unchanged offsets (a size-matched asset hot-reload). The offsets and
    /// counts above identify only *where* the geometry lives, so without this
    /// counter an in-place rewrite is indistinguishable from no change at all
    /// and a ray-tracing BLAS built over the previous contents would be reused.
    pub geometry_generation: u32,
    /// Column-major model-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Index into the shared texture pool for this object's albedo map.
    /// Clamped to the last slot if out of range.
    pub texture_slot: usize,
    /// Index into the shared texture pool for this object's normal map, or
    /// `NO_NORMAL_MAP_SLOT` when the object has no normal map (the backend then
    /// substitutes the flat-normal fallback).
    pub normal_map_slot: usize,
    /// Per-draw material scalars pushed to the fragment shader at buffer(3).
    pub material: MaterialUniforms,
    /// When false the object is skipped in both the shadow and main passes.
    /// Which world shader renders this object: the dense `ShaderHandle` value of
    /// the material's `shader` reference, or 0 (the world default) when the
    /// material names none. The backend binds the matching pipeline for each
    /// bucket; the value never changes after the draw is built.
    pub shader_bucket: u32,
    /// Controlled by scene switches to hide props belonging to inactive scenes.
    pub visible: bool,
    /// When false the object's geometry is not yet uploaded to the GPU buffers
    /// and the object is skipped in every pass. Always true unless the asset-
    /// streaming subsystem is active; the mesh streamer flips it to true once
    /// the geometry region is resident.
    /// Read by the shadow/main/velocity passes.
    pub resident: bool,
    /// World-space axis-aligned bounding box used for frustum culling.
    /// Baked from the mesh's local bounds and the prop's initial model matrix
    /// at graphics init. Set to a degenerate (NaN) box to disable culling.
    pub bb_min: [f32; 3],
    /// Upper corner of the world-space culling AABB.
    pub bb_max: [f32; 3],
    /// The mesh's own box, in model space, for a draw whose culling box
    /// follows its model: every [`Self::set_model`] re-derives `bb_min` /
    /// `bb_max` from it, so a draw moved after it was built (a planet world's
    /// frame moving under it) stays cullable. `None` keeps the world box as
    /// built.
    pub local_bounds: Option<([f32; 3], [f32; 3])>,
    /// If > 0, the object is skipped when the camera is further than this from
    /// the AABB center. Lets a Prop opt in to distance-based unloading.
    pub cull_distance: f32,
    /// LOD1..N slices for this draw, in ascending `switch_distance` order.
    /// Empty when the mesh declared `lod_levels <= 1`; the renderer then
    /// always uses the LOD0 `(index_offset, index_count)` carried on this
    /// object. With non-empty alternates the renderer picks the highest
    /// slice whose threshold is ≤ the camera→AABB-center distance each frame.
    pub lod_alternates: Vec<LodSlice>,
}

impl DrawObject {
    /// Pick the active `(index_offset, index_count)` for this object given
    /// the current camera distance to its AABB center. Returns the LOD0
    /// pair when no alternates are present or when the camera is closer
    /// than the first alternate's threshold; otherwise returns the
    /// highest-indexed alternate whose `switch_distance` ≤ `distance`.
    pub fn active_lod(&self, distance: f32) -> (usize, usize) {
        crate::gfx::lod::pick_lod_slice(
            (self.index_offset, self.index_count),
            &self.lod_alternates,
            distance,
        )
    }

    /// Returns true when the AABB encodes valid finite bounds suitable for
    /// frustum/distance culling. A degenerate box (NaN min) disables culling.
    pub fn cullable(&self) -> bool {
        crate::gfx::lod::bounds_finite(self.bb_min, self.bb_max)
    }

    /// Place the draw at `model`, its world box following when it carries
    /// [`Self::local_bounds`].
    pub fn set_model(&mut self, model: [[f32; 4]; 4]) {
        self.model = model;
        if let Some((lo, hi)) = self.local_bounds {
            (self.bb_min, self.bb_max) = crate::gfx::frustum::transform_aabb(lo, hi, model);
        }
    }
}

/// Per-object record consumed by the Metal "bindless" static main pass.
///
/// The static pipeline no longer rebinds model/material/textures per draw
/// call: every object's transform, material scalars, and texture-pool indices
/// live in one GPU buffer, and the vertex/fragment shaders index it by the
/// object id delivered through `[[base_instance]]`. That makes each draw call
/// stateless, which is the prerequisite for the GPU-driven compute-cull +
/// indirect-command-buffer pass.
///
/// The renderer rebuilds the buffer each frame from its `DrawObject` list, so
/// `update_model` / `update_visibility` changes are picked up automatically.
/// `bb_min` / `bb_max` / `cull_distance` mirror the `DrawObject` AABB and are
/// unused by the main-pass shaders; they are carried so a GPU-driven compute
/// cull kernel can read object bounds straight from this buffer without a
/// layout change.
///
/// Layout (208 bytes) must stay in sync with the shader-side record, declared
/// once in `shaders/object_common.hlsl` and spliced into every pass that
/// strides the buffer; `shader_layout` in concinnity-device pins this struct
/// against the compiled module's layout of it on every target.
///
/// `albedo_index` / `normal_index` (and the emissive / ORM indices)
/// are indices into each backend's single handle-indexed texture pool: albedo,
/// normal, and every optional map share one dense table, so a texture used in
/// more than one role resolves to one descriptor. The pool's two fallbacks
/// occupy reserved slots past the real textures: an object with no normal map
/// addresses the flat-normal entry and one with no albedo addresses the white
/// entry, so neither index needs a shader branch.
#[derive(Copy, Clone, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct GpuObjectData {
    /// Column-major model-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Linear-space RGB multiplier on the albedo sample.
    pub tint: [f32; 3],
    /// Perceptual roughness [0, 1].
    pub roughness: f32,
    /// Additive emission color in linear space.
    pub emissive: [f32; 3],
    /// Metallic factor [0, 1].
    pub metallic: f32,
    /// Index into the bindless texture pool for this object's albedo map.
    pub albedo_index: u32,
    /// Index into the bindless texture pool for this object's normal map.
    pub normal_index: u32,
    /// Bindless-pool index for the emissive map (0 = none). Mirrors
    /// `MaterialUniforms::emissive_map_index`.
    pub emissive_map_index: u32,
    /// Bindless-pool index for the packed occlusion/roughness/metalness map
    /// (0 = none). Mirrors `MaterialUniforms::orm_map_index`.
    pub orm_map_index: u32,
    /// World-space AABB minimum (compute-cull bounds input).
    pub bb_min: [f32; 3],
    /// View-distance cutoff; 0 = no cutoff (compute-cull bounds input).
    pub cull_distance: f32,
    /// World-space AABB maximum (compute-cull bounds input).
    pub bb_max: [f32; 3],
    /// Alpha-cutout threshold in [0, 1]; 0 disables the test. Mirrors
    /// `MaterialUniforms::alpha_cutoff`.
    pub alpha_cutoff: f32,
    /// Row of the material parameter table this object reads. Mirrors
    /// `MaterialUniforms::params_index`.
    pub params_index: u32,
    /// The far-field blend band, for a terrain under grass. Mirrors
    /// `MaterialUniforms::far_field`.
    pub far_field: FarFieldTint,
    /// How a planet tile's vertices morph toward its parent's mesh; all zero
    /// for every other draw.
    pub tile_morph: GpuTileMorph,
}

/// A planet tile's draw: its quadtree level and the sides it meets tiles of
/// other levels on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanetTileDraw {
    /// The tile's quadtree level.
    pub level: u8,
    /// Where it meets coarser or finer tiles.
    pub seams: crate::planet::TileSeams,
}

/// How a planet tile's vertices morph toward its parent's mesh, as its
/// object record carries it. 48 bytes, declared in `tile_morph_types.hlsl`
/// as two `float4`s and a `uint4`.
///
/// A vertex `d` meters from `eye` morphs from none at
/// [`crate::planet::MORPH_START`] of `range` to all of the way at `range`;
/// the motion vectors measure last frame's morph from `prev_eye`. A `range`
/// of 0 leaves every vertex where it is.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuTileMorph {
    /// The camera, in the simulated frame.
    pub eye: [f32; 3],
    /// How far from `eye` the tile's level reaches, in meters.
    pub range: f32,
    /// The camera last frame, in this frame's
    /// coordinates.
    pub prev_eye: [f32; 3],
    /// The sides beside a coarser tile in the low four bits (by
    /// [`crate::planet::TileSide::bit`]), beside finer tiles in the next four.
    pub seams: u32,
    /// The tile's first vertex in the shared vertex stream, where the morph
    /// reads its neighbors.
    pub vertex_base: u32,
    /// Keeps the row a whole number of `float4`s.
    pub _pad: [u32; 3],
}

// The world bound and per-object cull range a packed record carries. The
// main-pass shaders ignore both; a GPU-driven compute cull reads object bounds
// straight from this buffer, so every packer fills them.
#[derive(Clone, Copy)]
pub(crate) struct RecordBounds {
    pub bb_min: [f32; 3],
    pub bb_max: [f32; 3],
    pub cull_distance: f32,
}

impl GpuObjectData {
    // Assemble a record from the pieces every packer resolves differently
    // (transform, bounds, pool indices) plus the material block they all copy
    // verbatim. Adding a material field lands in one place instead of three.
    fn new(
        model: [[f32; 4]; 4],
        material: &MaterialUniforms,
        bounds: RecordBounds,
        albedo_index: u32,
        normal_index: u32,
    ) -> Self {
        Self {
            model,
            tint: material.tint,
            roughness: material.roughness,
            emissive: material.emissive,
            metallic: material.metallic,
            albedo_index,
            normal_index,
            emissive_map_index: material.emissive_map_index,
            orm_map_index: material.orm_map_index,
            bb_min: bounds.bb_min,
            cull_distance: bounds.cull_distance,
            bb_max: bounds.bb_max,
            alpha_cutoff: material.alpha_cutoff,
            params_index: material.params_index,
            far_field: material.far_field,
            tile_morph: GpuTileMorph::default(),
        }
    }
}

/// Pack one `DrawObject` into its `GpuObjectData` record for the DirectX and
/// Vulkan bindless static pass. The caller supplies the resolved texture-pool
/// indices: `albedo_index` / `normal_index` are dense indices into the one
/// handle-indexed pool (a normal-less object's `normal_index` points at the
/// pool's flat-normal fallback slot), identical across all three backends.
///
/// `bb_min` / `bb_max` / `cull_distance` are copied even though the
/// main-pass shaders ignore them, so a GPU-driven compute cull can read object
/// bounds straight from this buffer without a layout change.
pub fn pack_object_record(obj: &DrawObject, albedo_index: u32, normal_index: u32) -> GpuObjectData {
    let bounds = RecordBounds {
        bb_min: obj.bb_min,
        bb_max: obj.bb_max,
        cull_distance: obj.cull_distance,
    };
    GpuObjectData::new(obj.model, &obj.material, bounds, albedo_index, normal_index)
}

/// Pack one instance of an `InstancedCluster` into a `GpuObjectData` for the
/// GPU-driven bindless instanced path: the cluster's material + flat-pool texture
/// indices, this instance's model matrix, and the instance's world AABB (the
/// cluster's mesh-local AABB transformed by the model) so the compute cull tests
/// each instance independently. Mirrors `pack_object_record`, sourcing the bounds
/// from the cluster rather than a `DrawObject`. The instanced mesh's indices are
/// already absolute (rebased at `append_mesh`), so the per-instance draw args use
/// `base_vertex = 0`.
pub fn pack_instance_record(
    cluster: &InstancedCluster,
    model: [[f32; 4]; 4],
    albedo_index: u32,
    normal_index: u32,
) -> GpuObjectData {
    let (bb_min, bb_max) =
        crate::gfx::frustum::transform_aabb(cluster.local_bb_min, cluster.local_bb_max, model);
    let bounds = RecordBounds {
        bb_min,
        bb_max,
        cull_distance: cluster.cull_distance,
    };
    GpuObjectData::new(model, &cluster.material, bounds, albedo_index, normal_index)
}

/// Expand every instance of every cluster into a flat `GpuObjectData` list for the
/// GPU-driven bindless instanced path, in cluster-then-instance order (so a
/// parallel per-instance draw-args list can be walked in the same order). The
/// texture-pool indices are resolved through `albedo_pool_index` /
/// `normal_pool_index`, matching `build_object_buffer`'s static addressing.
/// `texture_count` is the pool's real-texture count (the flat-normal fallback
/// sits at `texture_count`).
pub fn instance_object_records(
    clusters: &[InstancedCluster],
    texture_count: u32,
) -> Vec<GpuObjectData> {
    let total: usize = clusters.iter().map(|c| c.instances.len()).sum();
    let mut records = Vec::with_capacity(total);
    for cluster in clusters {
        let albedo = albedo_pool_index(cluster.texture_slot, texture_count);
        let normal = normal_pool_index(cluster.normal_map_slot, texture_count);
        for &model in &cluster.instances {
            records.push(pack_instance_record(cluster, model, albedo, normal));
        }
    }
    records
}

// Factor the skinned mesh's bind-pose AABB half-extents are scaled by before the
// box is transformed into the per-frame world bound the GPU culler tests. Skinned
// geometry deforms every frame, so the bind-pose box does not cover poses that
// reach past it (raised arms, a wide stance); padding keeps those from being
// wrongly culled. Conservative -- loose, but never a false negative. A tight
// per-frame bound emitted by the skin compute kernel is a planned follow-up.
const SKINNED_BB_PAD_FACTOR: f32 = 2.0;

/// Build the GPU-driven cull/draw record for one skinned object, folded into the
/// bindless main pass as rigid geometry. The skin compute kernel deforms the
/// bind-pose vertices into MODEL space, so `model` is applied after skinning
/// (model -> world) exactly like a static object; the cull bound is the padded
/// bind-pose AABB transformed by `model`. Mirrors `pack_instance_record`, sourcing
/// material + flat texture indices from the skinned object.
pub fn pack_skinned_record(
    obj: &SkinnedDrawObject,
    albedo_index: u32,
    normal_index: u32,
) -> GpuObjectData {
    let mut lo = [0.0f32; 3];
    let mut hi = [0.0f32; 3];
    for a in 0..3 {
        let center = 0.5 * (obj.local_bb_min[a] + obj.local_bb_max[a]);
        let half = 0.5 * (obj.local_bb_max[a] - obj.local_bb_min[a]) * SKINNED_BB_PAD_FACTOR;
        lo[a] = center - half;
        hi[a] = center + half;
    }
    let (bb_min, bb_max) = crate::gfx::frustum::transform_aabb(lo, hi, obj.model);
    let bounds = RecordBounds {
        bb_min,
        bb_max,
        cull_distance: 0.0,
    };
    GpuObjectData::new(obj.model, &obj.material, bounds, albedo_index, normal_index)
}

/// Per-object draw parameters consumed by the GPU-driven compute cull kernel.
/// Parallel to `GpuObjectData` (one record per `DrawObject`, same index):
/// `GpuObjectData` carries the cull bounds the kernel tests, this carries the
/// indexed-draw arguments the kernel encodes into the indirect command buffer
/// (Metal `MTLIndirectCommandBuffer` / DirectX `ExecuteIndirect` argument
/// buffer / Vulkan `multiDrawIndexedIndirect` buffer) when an object survives
/// the cull.
///
/// The kernel reads `flags` to decide an object's fate without re-deriving it
/// from the AABB: `ENABLED` gates the per-frame `visible` / `resident` state,
/// `CULLABLE` selects whether the object is frustum/distance-tested at all
/// (mirrors `DrawObject::cullable()`: non-cullable objects always draw).
///
/// Layout (16 bytes) must stay in sync with the `GpuDrawArgs` struct in
/// `cull.hlsl` and in the Metal ICB encode kernel, `cull_encode.metal`.
#[derive(Copy, Clone, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct GpuDrawArgs {
    /// Number of indices to draw (`DrawObject::index_count`).
    pub index_count: u32,
    /// Element offset into the shared index buffer (`DrawObject::index_offset`).
    pub index_offset: u32,
    /// Value added to every index before the vertex fetch
    /// (`DrawObject::base_vertex`; always >= 0 in practice).
    pub base_vertex: u32,
    /// Cull-decision bits; see `DrawArgsFlags`.
    pub flags: u32,
}

// Bit flags packed into `GpuDrawArgs::flags`.
pub(crate) struct DrawArgsFlags;

impl DrawArgsFlags {
    // The object is visible and its geometry is resident this frame. When
    // clear the kernel resets the object's indirect command (draws nothing).
    pub(crate) const ENABLED: u32 = 1;
    // The object has a finite AABB and should be frustum/distance-culled.
    // When clear the object always draws (subject to `ENABLED`).
    pub(crate) const CULLABLE: u32 = 2;
    // The record's entry in the model-history ring was written for a different
    // occupant, so the G-buffer pre-pass must reproject through the current
    // model instead. Mirrored by DRAW_NO_HISTORY in object_common.hlsl.
    pub(crate) const NO_HISTORY: u32 = 4;
    // The record's shader bucket rides bits 8..16: the cull kernel routes the
    // record's indirect command into that bucket's ICB. Values and layout are
    // mirrored by the cull shaders (DRAW_BUCKET_SHIFT in cull.hlsl and
    // cull_encode.metal).
    pub(crate) const BUCKET_SHIFT: u32 = 8;
}

/// Upper bound on the Shaders a world's buckets are budgeted for, the default
/// and its grass twins (counted as one) included. Enforced by the world-shape
/// check and quoted as a number in the Shader asset docs.
pub const MAX_SHADER_BUCKETS: usize = 8;

/// Upper bound on shader buckets a cull dispatch can route between (the ICB
/// argument-buffer array length in the Metal encode kernel): the budgeted
/// Shaders plus the default's second grass twin. Mirrored by `MAX_DRAW_BUCKETS`
/// in `cull_encode.metal`.
pub const MAX_DRAW_BUCKETS: usize = MAX_SHADER_BUCKETS + 1;

/// Pack the per-frame cull-decision bits for one `DrawObject`. Mirrors the
/// CPU `visible = BVH(cullable) + always_draw` partition: an object draws when
/// it is visible and resident, and is frustum-tested only when it has finite
/// bounds.
pub fn draw_args_flags(visible: bool, resident: bool, cullable: bool) -> u32 {
    let mut flags = 0;
    if visible && resident {
        flags |= DrawArgsFlags::ENABLED;
    }
    if cullable {
        flags |= DrawArgsFlags::CULLABLE;
    }
    flags
}

/// The `GpuDrawArgs::flags` bit that sends the G-buffer pre-pass to a record's
/// current model rather than its model-history entry. Set for a record whose
/// history entry was written for a different occupant, and for every record
/// while no consumer reads motion (which collapses the motion vector to zero,
/// since the pre-pass is handed `prev_vp == cur_vp` then).
pub fn draw_args_no_history() -> u32 {
    DrawArgsFlags::NO_HISTORY
}

/// Pack a `DrawObject`'s shader bucket into the `GpuDrawArgs::flags` upper bits,
/// clamped to the routing budget the cull kernels implement. A bucket past the
/// budget renders with the world default program rather than reading a command
/// region that does not exist; the world-shape check fails such a world at build
/// time, so the clamp is a backstop.
pub fn draw_args_bucket_bits(shader_bucket: u32) -> u32 {
    let bucket = u32::saturating_from((shader_bucket as usize).min(MAX_DRAW_BUCKETS - 1));
    bucket << DrawArgsFlags::BUCKET_SHIFT
}

/// One GPU-instanced draw: a shared mesh slice + material rendered at many
/// world-space transforms in a single `drawIndexedInstanced` / `cmd_draw_indexed`
/// (`instance_count` > 1). The cluster has a single union AABB across all
/// instances; it is frustum-tested as a whole. Cluster culling trades per-
/// instance precision for one draw call instead of N.
pub struct InstancedCluster {
    /// Byte offset into the shared vertex buffer. Currently unused because
    /// every backend keeps the shared vertex buffer bound at offset 0 and
    /// `append_mesh` rebases indices accordingly; retained for symmetry with
    /// `DrawObject` and future per-cluster vertex bindings.
    pub vertex_offset: usize,
    /// Number of vertices this cluster's mesh occupies in the shared vertex
    /// buffer, starting at `vertex_offset`. Used by the shrinkable-seed
    /// compaction to relocate the cluster's geometry region when streamed-mesh
    /// gaps are removed from the shared buffer.
    pub vertex_count: usize,
    /// Element offset into the shared index buffer.
    pub index_offset: usize,
    /// Number of indices per instance.
    pub index_count: usize,
    /// Index into the shared texture pool for this cluster's albedo map.
    pub texture_slot: usize,
    /// Index into the shared texture pool for this cluster's normal map, or
    /// `NO_NORMAL_MAP_SLOT` when the cluster has no normal map.
    pub normal_map_slot: usize,
    /// Per-cluster material scalars, identical for every instance.
    pub material: MaterialUniforms,
    /// Union AABB of all per-instance world AABBs; used for cluster-wide
    /// frustum culling. A degenerate (NaN) box disables culling.
    pub cluster_bb_min: [f32; 3],
    /// Upper corner of the cluster's union world AABB.
    pub cluster_bb_max: [f32; 3],
    /// Mesh-local (object-space) AABB, identical for every instance. The
    /// GPU-driven instanced path transforms this by each instance's model matrix
    /// to get a per-instance world AABB for per-instance frustum/distance/Hi-Z
    /// culling; the union AABB above is only the whole-cluster bound.
    pub local_bb_min: [f32; 3],
    /// Upper corner of the mesh-local bind AABB.
    pub local_bb_max: [f32; 3],
    /// View-distance cutoff applied to the cluster center. 0 = no cutoff.
    pub cull_distance: f32,
    /// Per-instance column-major model matrices. Uploaded to a transient GPU
    /// buffer each frame.
    pub instances: Vec<[[f32; 4]; 4]>,
    /// LOD1..N slices for this cluster's mesh, in ascending `switch_distance`
    /// order. Empty when the mesh declared `lod_levels <= 1`. Each per-instance
    /// LOD is picked from camera distance to that instance's translation and
    /// written into its cull record (see `lod::for_each_instance_lod`). The
    /// vertex set is shared with LOD0 (QEM decimation preserves vertices), so
    /// only `(index_offset, index_count)` varies per slice.
    pub lod_alternates: Vec<LodSlice>,
}

impl InstancedCluster {
    /// True when the cluster AABB encodes valid finite bounds suitable for
    /// frustum / distance culling.
    pub fn cullable(&self) -> bool {
        crate::gfx::lod::bounds_finite(self.cluster_bb_min, self.cluster_bb_max)
    }
}

/// One skeletally animated draw: a slice of the shared skinned vertex/index
/// buffers plus the per-joint matrix buffer the vertex shader blends.
///
/// Skinned objects deform every frame, so they are excluded from the static
/// BVH and drawn unconditionally (after the visibility flag). The joint
/// matrices live in a separate per-object GPU buffer the renderer rewrites
/// each frame from `AnimationSystem`'s output.
pub struct SkinnedDrawObject {
    /// Vertex offset (in vertex units, not bytes) of this slot's first
    /// vertex in the shared skinned vertex buffer. The shared index buffer
    /// holds absolute vertex indices (each slot's mesh-relative indices are
    /// re-based onto this `vertex_base` at upload time), so the value is
    /// not consumed by the draw call itself, but the asset hot-reload's
    /// `rebuild_skinned_geometry` path needs to know each slot's vertex
    /// region to copy unchanged geometry across a size-changing rebuild.
    pub vertex_base: u32,
    /// Number of vertices in this slot's region of the shared skinned
    /// vertex buffer.
    pub vertex_count: usize,
    /// Element offset into the shared skinned index buffer.
    pub index_offset: usize,
    /// Number of indices to draw.
    pub index_count: usize,
    /// Column-major model-to-world matrix. Applied after skinning.
    pub model: [[f32; 4]; 4],
    /// Index into the shared texture pool for this object's albedo map.
    pub texture_slot: usize,
    /// Index into the shared texture pool for this object's normal map, or
    /// `NO_NORMAL_MAP_SLOT` when the object has no normal map.
    pub normal_map_slot: usize,
    /// Per-object material scalars.
    pub material: MaterialUniforms,
    /// When false the object is skipped in both the shadow and main passes.
    pub visible: bool,
    /// Number of joints in this object's skeleton, capped at `MAX_JOINTS`.
    /// The vertex shader only blends matrices below this count.
    pub joint_count: usize,
    /// Mesh-local (object-space) bind-pose AABB, computed from the bind-pose
    /// vertices at load. The GPU-driven skinned fold pads this (animations reach
    /// past the bind pose) and transforms it by `model` to get the per-frame world
    /// bound the cull kernel frustum/Hi-Z tests. See `pack_skinned_record`.
    pub local_bb_min: [f32; 3],
    /// Upper corner of the mesh-local bind-pose AABB.
    pub local_bb_max: [f32; 3],
    /// LOD1..N slices, in ascending `switch_distance` order. Empty when
    /// the `SkinnedMesh` declared `lod_levels <= 1`. Each slice points at a
    /// rebased index range in the shared skinned IB; QEM half-edge
    /// decimation keeps the vertex set unchanged so all LODs share this
    /// slot's `vertex_base` / `vertex_count`.
    pub lod_alternates: Vec<LodSlice>,
}

impl SkinnedDrawObject {
    /// Pick the active `(index_offset, index_count)` for this object given
    /// the camera distance to its model-matrix translation. Returns the
    /// LOD0 pair when no alternates are present or the distance is below
    /// the first threshold.
    pub fn active_lod(&self, distance: f32) -> (usize, usize) {
        crate::gfx::lod::pick_lod_slice(
            (self.index_offset, self.index_count),
            &self.lod_alternates,
            distance,
        )
    }

    /// World-space translation of this object (column 3 of the model
    /// matrix). Skinned objects have no static AABB (they deform every
    /// frame), so per-frame LOD picks use the model translation as the
    /// stand-in for an AABB center.
    pub fn translation(&self) -> [f32; 3] {
        [self.model[3][0], self.model[3][1], self.model[3][2]]
    }
}

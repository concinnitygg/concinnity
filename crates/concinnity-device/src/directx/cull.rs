//! Compute-driven GPU culling: one thread per build-time `DrawObject`
//! frustum/distance-tests the object's `GpuObjectData` AABB against the six
//! CPU-extracted frustum planes, optionally Hi-Z-occlusion-tests it against the
//! previous frame's depth pyramid (see `directx/hiz.rs`), and counts the ones
//! it keeps per stream; `directx/compact.rs` then packs their `ExecuteIndirect`
//! commands into one list per stream in the per-frame argument buffer, with
//! the list's length behind the commands. The main bindless pass issues each
//! list with one `ExecuteIndirect` by that count, so the CPU never walks the
//! draw list.
//!
//! The frustum and distance maths mirror `gfx::frustum` exactly (the six
//! planes are extracted CPU-side already normalized) so the GPU path culls
//! identically to the CPU BVH path it replaces. `GpuObjectData` / `GpuDrawArgs`
//! mirror `gfx::render_types`; `IndirectCommand` is a b0 root constant (the
//! object id) followed by `D3D12_DRAW_INDEXED_ARGUMENTS`, matching the command
//! signature built by `create_cull_command_signature`. Mirrors src/metal/cull.rs.

use concinnity_core::fanout::FanOutTiming;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::cull_status::CullStatus;
use concinnity_engine::gfx::frustum::{Frustum, Plane};
use concinnity_engine::gfx::lod;
use concinnity_engine::gfx::render_types::CaptureTiles;
use concinnity_engine::gfx::render_types::{
    GpuCompactParams, MAX_SHADOWED_SPOTS, StreamLayout, culled_records,
};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::cull::CULL_GROUP;
use concinnity_engine::render::shadow::far::far_caster_radius;
use concinnity_engine::render::temporal::model_history::HistoryMode;
use concinnity_engine::render::uniforms::cull::{CullHiz, CullParams, CullView, ShadowCullParams};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::allocator::PooledBuffer;
use crate::directx::com;
use crate::directx::compact::{CompactPass, DrawLists, UavSpan};
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::pso::compute_pso;
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::transition_barrier;

// The compute cull kernel and its root signature, plus two-pass occlusion's
// late kernel over the same signature (`None` when two-pass is off).
pub(in crate::directx) struct CullKernels {
    pub root_sig: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
    pub pso_late: Option<ID3D12PipelineState>,
}

// GPU-driven cull + main pass. A compute kernel frustum/distance-tests every
// record and writes one `ExecuteIndirect` command per object; the main pass
// issues each bucket's region with one `ExecuteIndirect`. All `Some`/non-empty
// only when the world has anything to drive.
pub(in crate::directx) struct CullState {
    // Packs the static draws' records into the object and draw-args buffers.
    pub packer: crate::record_pack::RecordPacker,
    // The main pass's root signature and bucket 0's PSO: the world default
    // Shader's pair where the world declares one, the engine's pair otherwise.
    pub main_bindless_root_sig: Option<ID3D12RootSignature>,
    pub main_bindless_pso: Option<ID3D12PipelineState>,
    // The G-buffer pre-pass's root signature, the cull command signature rebuilt
    // against it, and bucket 0's pre-pass PSO, present once a G-buffer exists. The
    // pre-pass reuses the main pass's indirect buffers.
    pub prepass_root_sig: Option<ID3D12RootSignature>,
    pub prepass_cmd_sig: Option<ID3D12CommandSignature>,
    pub main_prepass_pso: Option<ID3D12PipelineState>,
    // Material-referenced world shader pipelines, indexed by `shader_bucket - 1`,
    // for the main pass and the pre-pass. `None` marks a bucket whose Shader is
    // not resident yet; both passes skip its draws.
    pub world_pipelines: concinnity_engine::render::shader::world_pipelines::WorldPipelines<
        super::init::pipelines::BucketPsos,
    >,
    // Commands reserved per single-stream region (a probe face, a spot slice),
    // fixed at init to the record capacity the buffers were sized for.
    pub bucket_stride: usize,
    // The routes and the bucket regions of the indirect buffers, fixed at
    // init: each bucket's region holds the records it can take.
    pub variants: concinnity_engine::gfx::render_types::GpuCullVariants,
    // Per-slot upload copies of `variants` with that slot's view, bound at
    // root param [6] of every view cull, persistently mapped.
    pub variant_buffers: Vec<PooledBuffer>,
    pub variant_ptrs: Vec<*mut u8>,
    // Per slot, the buckets its view can pick, a bit each (see
    // `GpuCullVariants::live_buckets`); the others' regions skip executing.
    pub live_buckets: Vec<std::sync::atomic::AtomicU64>,
    // The engine's compiled bindless main-pass stages, retained so a shader
    // bucket warmed mid-session can build its pipeline without recompiling the
    // HLSL. See [`super::init::pipelines::BindlessMainShaders`].
    pub bindless_main_shaders: super::init::pipelines::BindlessMainShaders,
    // Per-frame `StructuredBuffer<GpuObjectData>` upload buffers, one per
    // frame-in-flight, persistently mapped. Rebuilt each frame.
    pub object_buffer_resources: Vec<PooledBuffer>,
    pub object_buffer_ptrs: Vec<*mut u8>,
    // The material parameter table, over the same slots as the object buffers;
    // `None` when the world has nothing to drive.
    pub material_params: Option<super::material_params::MaterialParams>,
    // One flat-pool SRV base per frame in flight, bound at root [5], so a streamed
    // swap rewrites the copy whose frame just fence-waited instead of draining.
    pub bindless_pool_gpu: Vec<SrvSlot>,
    pub cull_kernels: Option<CullKernels>,
    pub cull_command_signature: Option<ID3D12CommandSignature>,
    // Per-frame `StructuredBuffer<GpuDrawArgs>` upload buffers (indexed-draw
    // args + per-frame cull-decision bits the kernel reads).
    pub draw_args_buffer_resources: Vec<PooledBuffer>,
    pub draw_args_buffer_ptrs: Vec<*mut u8>,
    // Per-frame indirect-command buffers the cull packs its lists into (UAV)
    // and the main pass consumes (`ExecuteIndirect`). Resting
    // `INDIRECT_ARGUMENT`. Each holds `indirect_commands` commands, every
    // bucket region's back to back, then the lists' counts.
    pub indirect_cmd_buffers: Vec<ID3D12Resource>,
    pub indirect_commands: usize,
    // The scan and scatter that pack each cull's kept draws per stream, and
    // each slot's group-count scratch (the late pass reusing the early pass's once its
    // packing has read it). Resting `UNORDERED_ACCESS`.
    pub compact: Option<super::compact::CompactKernels>,
    pub group_count_buffers: Vec<ID3D12Resource>,
    // Per-frame per-object cull-status buffers (one u32 each): early writes,
    // late reads. Resting `UNORDERED_ACCESS`.
    pub cull_status_buffers: Vec<ID3D12Resource>,
    // Per-frame second indirect-command buffers for two-pass occlusion.
    pub indirect_cmd_buffers_late: Vec<ID3D12Resource>,
    // GPU-driven shadows: the depth-only pipeline, its command signature, and
    // per-frame views with a region per cascade.
    pub shadow_bindless_root_sig: Option<ID3D12RootSignature>,
    pub shadow_bindless_pso: Option<ID3D12PipelineState>,
    pub shadow_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    // Frustum-only shadow cull kernel (`main_shadow`), shares the cull root sig.
    pub cull_pso_shadow: Option<ID3D12PipelineState>,
    pub shadow_views: Vec<ShadowCullViews>,
    // The spot shadow pass's views, one per frame-in-flight with a packed
    // region per spot slice, written by the same shadow kernel. Its own, so the
    // spot pass shares no state with the cascade pass. Empty when the world has
    // no shadowed spot.
    pub spot_views: Vec<ShadowCullViews>,
    // The far shadow's D16 pipeline and its views, one per frame-in-flight
    // with one region, written by the same shadow kernel. Empty without a far
    // map.
    pub far_shadow_bindless_pso: Option<ID3D12PipelineState>,
    pub far_views: Vec<ShadowCullViews>,
    // Per-frame model-history buffers (one column-major `float4x4` per cull
    // record), device-local: only the snapshot kernel writes them and only the
    // pre-pass reads them, so the host never touches their bytes. Frame `R`
    // reads the slot frame `R - 1` filled.
    pub prev_model_buffers: Vec<ID3D12Resource>,
    // The snapshot kernel that fills them from the object buffer.
    pub model_history_root_sig: Option<ID3D12RootSignature>,
    pub model_history_pso: Option<ID3D12PipelineState>,
    // `Culling.two_pass_occlusion`, as requested by the world.
    pub occlusion_two_pass: bool,
    // Hi-Z (depth-mip pyramid) used by the cull kernel for occlusion culling.
    // Built each frame after `execute_graph`; consumed by the *next* frame's
    // cull dispatch through `prev_view_proj`.
    pub hiz: Option<super::hiz::HiZResources>,
}

// Byte stride of one `IndirectCommand` in the cull kernel's output buffer: a
// 1-DWORD object-id root constant + `D3D12_DRAW_INDEXED_ARGUMENTS` (5 DWORDs).
pub(in crate::directx) const INDIRECT_COMMAND_STRIDE: u32 = 24;

// Pipeline + command signature builders

// Root signature for the cull kernels: `CullParams` constants, records and
// draw args as root SRVs, the Hi-Z pyramid in a table (root SRVs cannot carry
// textures), group counts and cull status as root UAVs, and the view's
// variants. The early cull writes `cull_status`; the late cull reads it.
pub(in crate::directx) fn create_cull_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use Visibility::All;
    use concinnity_engine::render::shader::programs::abi::cull::dx as cull;
    RootSig::new()
        .constants::<CullParams>(cull::CULL, All) // [0]
        .srv(cull::OBJECTS, All) // [1]
        .srv(cull::DRAW_ARGS, All) // [2]
        .srv_table(cull::HIZ, 1, All) // [3] every mip
        .uav(cull::GROUP_COUNTS, All) // [4]
        .uav(cull::CULL_STATUS, All) // [5]
        .srv(cull::VARIANTS, All) // [6]
        .build(device, "cull root sig")
}

// Compute pipeline state for the GPU-cull kernel.
pub(in crate::directx) fn create_cull_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(device, root_sig, cs, "cull")
}

// Command signature for the main pass's `ExecuteIndirect`: the b0 object-id
// constant, then `DrawIndexed`, matching the kernel's `IndirectCommand`.
// Created against the bindless main root signature it binds into.
pub(in crate::directx) fn create_cull_command_signature(
    device: &ID3D12Device,
    bindless_root_sig: &ID3D12RootSignature,
) -> RenderResult<ID3D12CommandSignature> {
    let arg_descs = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: 0,
                    DestOffsetIn32BitValues: 0,
                    Num32BitValuesToSet: 1,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: INDIRECT_COMMAND_STRIDE,
        NumArgumentDescs: u32::saturating_from(arg_descs.len()),
        pArgumentDescs: arg_descs.as_ptr(),
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateCommandSignature(&raw const desc, bindless_root_sig, &raw mut sig) }
        .map_err(|e| map_hresult(e.code(), "create cull command signature"))?;
    sig.ok_or_else(|| {
        RenderError::Other("create cull command signature: returned None".to_string())
    })
}

// One cull into a region of an indirect buffer: the region, the frustum, the
// eye for LOD (and distance), the records `[first, end)` to leave out, and the
// coarse grid planet tiles pack through.
#[derive(Clone, Copy)]
pub(in crate::directx) struct RegionCull {
    pub region: usize,
    pub frustum: Frustum,
    pub eye: [f32; 3],
    pub exclude: [u32; 2],
    pub capture_tiles: Option<CaptureTiles>,
    // The smallest caster a shadow view keeps; 0 keeps every size, and a
    // camera view's kernel never reads it.
    pub min_caster_radius: f32,
}

impl RegionCull {
    // A placeholder for the unused tail of a fixed-capacity list.
    pub(in crate::directx) const EMPTY: Self = Self {
        region: 0,
        frustum: Frustum {
            planes: [Plane {
                normal: [0.0; 3],
                d: 0.0,
            }; 6],
        },
        eye: [0.0; 3],
        exclude: [0, 0],
        capture_tiles: None,
        min_caster_radius: 0.0,
    };
}

// The view block a run of region culls' kernel reads at b0.
#[derive(Clone, Copy)]
enum RegionKernel {
    // The camera cull: frustum, eye, exclusion, and the Hi-Z half.
    Camera,
    // The shadow cull: a light's frustum and the smallest caster it keeps.
    Shadow,
}

// The kernel and output buffer a run of region culls shares. Region `r`'s
// commands start at command `r * region_stride`, or with `packing` at
// `r * packing.lists_stride` bytes, its lists there.
#[derive(Clone, Copy)]
struct RegionCullPass<'a> {
    pso: &'a ID3D12PipelineState,
    kernel: RegionKernel,
    indirect: &'a ID3D12Resource,
    status_gva: u64,
    region_stride: usize,
    packing: Option<RegionPacking<'a>>,
}

// Where a run of region culls packs each region's kept draws: its own
// group-count scratch, status and lists, each `*_stride` bytes past the last
// region's.
#[derive(Clone, Copy)]
pub(in crate::directx) struct RegionPacking<'a> {
    pub(in crate::directx) group_counts: &'a ID3D12Resource,
    pub(in crate::directx) group_stride: u64,
    pub(in crate::directx) status: &'a ID3D12Resource,
    pub(in crate::directx) status_stride: u64,
    pub(in crate::directx) lists_stride: u64,
}

// One frame's shadow views of one pass, a region of `bucket_stride` records
// each: their lists, status and group counts. Resting in COMMON until first
// used, then the lists in INDIRECT_ARGUMENT and the rest in UNORDERED_ACCESS.
pub(in crate::directx) struct ShadowCullViews {
    pub lists: ID3D12Resource,
    pub status: ID3D12Resource,
    pub group_counts: ID3D12Resource,
}

// Bytes of one shadow view's status: a word per record.
pub(in crate::directx) fn shadow_status_bytes(n_cull: usize) -> u64 {
    crate::directx::context::align256((n_cull * size_of::<u32>()) as u64).max(256)
}

// Per-frame buffer fill + encoder

impl DxContext {
    // Write slot `slot`'s variant block: the routes and bucket regions, seen
    // from the main view, where the grass hands over to the canopy and under
    // the view's planet sky; and note which buckets that view can pick.
    fn write_cull_variants(&self, slot: usize) {
        let Some(&ptr) = self.cull.variant_ptrs.get(slot) else {
            return;
        };
        let mut variants = self.cull.variants;
        variants.set_view(
            self.grass_lod.get().map_or(0.0, |l| l.fade_start()),
            &self.state.view.daylight.variant_sky(),
        );
        if let Some(live) = self.cull.live_buckets.get(slot) {
            live.store(
                variants.live_buckets(
                    self.shader_bucket_count(),
                    &self.state.view.daylight.variant_sky(),
                ),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let bytes = bytemuck::bytes_of(&variants);
        // SAFETY: the slot's buffer was mapped at init with room for one block, and its
        // fence has signaled (or it is the probe bake's reserved slot before its capture is
        // submitted), so the GPU does not read it while this copy runs.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    }

    // The variant block slot `slot`'s view culls read, at root param [6].
    fn bind_cull_variants(&self, cmd: &ID3D12GraphicsCommandList, slot: usize) {
        if let Some(buffer) = self.cull.variant_buffers.get(slot) {
            // SAFETY: the command list is recording, and the slot's variant buffer is live and
            // CPU-written before the frame's submit.
            unsafe { cmd.SetComputeRootShaderResourceView(6, com::gpu_va(buffer)) };
        }
    }

    // Shader-bucket regions the cull kernel routes between: the world default
    // program plus one per material-referenced world shader. 1 when the world
    // declares no extra shaders, which collapses the indirect buffer to the
    // single region every pass used before buckets existed.
    pub(in crate::directx) fn shader_bucket_count(&self) -> usize {
        self.cull.world_pipelines.bucket_count()
    }

    // The main PSO a probe or planar face draws its one command stream under:
    // the default's fullest grass variant when the world has one, so a far
    // band seen in a face keeps its canopy, else the default's.
    pub(in crate::directx) fn face_bindless_pso(&self) -> Option<&ID3D12PipelineState> {
        let superset = self
            .cull
            .world_pipelines
            .superset_bucket()
            .and_then(|bucket| self.cull.world_pipelines.get(bucket as usize));
        match superset {
            Some(psos) => Some(&psos.main),
            None => self.cull.main_bindless_pso.as_ref(),
        }
    }

    // Pack ring slot `slot`'s object and draw-arg records, each draw's LOD picked
    // by distance from `cam_pos`. Static draws pack across the job pool when that
    // pays; the runtime reserve and skinned tail follow, and the instance tail only
    // has its LOD patched. Returns the static packing time.
    pub(in crate::directx) fn build_record_buffers(
        &self,
        slot: usize,
        cam_pos: [f32; 3],
        history: HistoryMode,
    ) -> FanOutTiming {
        use concinnity_engine::gfx::render_types::{
            GpuDrawArgs, GpuObjectData, albedo_pool_index, draw_args_flags, normal_pool_index,
            pack_skinned_record,
        };
        use concinnity_engine::render::record_pack::{PackView, StaticPack, pool_object_record};
        let (Some(&objects), Some(&args)) = (
            self.cull.object_buffer_ptrs.get(slot),
            self.cull.draw_args_buffer_ptrs.get(slot),
        ) else {
            return FanOutTiming::default();
        };
        let n_cull = self.state.draw.record_count();
        // SAFETY: both buffers were mapped at init with room for `draw.record_count()`
        // records, which never changes; slot `slot` is this frame's (its fence has
        // signaled) or the probe bake's reserved slot before its capture is
        // submitted, so the GPU reads neither, and these are the only views of
        // either mapping until this function returns.
        let records = unsafe { mapped_records::<GpuObjectData>(objects, n_cull) };
        // SAFETY: as above, for the draw-args mapping.
        let draw_args = unsafe { mapped_records::<GpuDrawArgs>(args, n_cull) };
        let (Some(records), Some(draw_args)) = (records, draw_args) else {
            tracing::error!("directx: cull record buffers are not mapped for {n_cull} records");
            return FanOutTiming::default();
        };
        // Shared handle-indexed pool indices, identical to Vulkan/Metal: albedo =
        // texture_slot, normal = the normal map's own handle (or the flat-normal
        // fallback slot for a normal-less draw). The bindless main pass + RT hit
        // shader bind the pool base, so a shared texture resolves to one descriptor.
        let texture_count = u32::saturating_from(self.scene.textures.len());
        // Main thread only, ahead of the encode fan-out: no worker borrows the tracker.
        let mut model_history = self.state.model_history.borrow_mut();
        model_history.begin(history, n_cull);
        // A see-through glass mesh (Layer 2) is disabled in the opaque pass when
        // the RT path is live: it draws in the transparent pass instead.
        let mesh_glass_active = self.mesh_glass_active();
        let view = PackView {
            cam_pos,
            mesh_glass_active,
            horizon: self
                .state
                .horizon_occluder
                .and_then(|o| o.seen_from(cam_pos)),
            routes: self.cull.world_pipelines.routes(),
        };

        let n_static = self
            .state
            .draw
            .n_objects
            .min(self.state.draw.objects.len())
            .min(n_cull);
        let (static_records, _) = records.split_at_mut(n_static);
        let (static_args, _) = draw_args.split_at_mut(n_static);
        let packing = self.cull.packer.pack(
            StaticPack {
                objects: &self.state.draw.objects[..n_static],
                records: static_records,
                args: static_args,
                history: model_history.window(),
            },
            &view,
            &|obj| pool_object_record(obj, texture_count),
        );

        // Runtime objects (streamed chunks, spawned clones), one record each in the
        // reserved region, drawn with the static prefix. They are not cullable (NaN
        // AABB), so a resident one always draws and a freed slot is disabled.
        let runtime_base = self.state.draw.runtime_record_base();
        let routes = self.cull.world_pipelines.routes();
        let n_resident_runtime = self.state.draw.for_each_runtime_record(|k, i, obj| {
            let Some((rec, arg)) = records
                .get_mut(runtime_base + k)
                .zip(draw_args.get_mut(runtime_base + k))
            else {
                return;
            };
            *rec = pool_object_record(obj, texture_count);
            rec.tile_morph = concinnity_engine::render::record_pack::object_tile_morph(obj);
            // `camera_distance` falls back to the model translation for a
            // non-cullable object, so the LOD pick works off a NaN AABB. Chunks
            // carry no alternates and land on LOD0 either way; a clone inherits
            // its template's.
            let d = lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            // A clone copies its template's material, so a see-through one
            // leaves the opaque pass the way a static one does.
            let opaque_visible =
                obj.visible && !(mesh_glass_active && obj.material.see_through != 0);
            *arg = GpuDrawArgs {
                index_count: u32::saturating_from(index_count),
                index_offset: u32::saturating_from(index_offset),
                base_vertex: obj.base_vertex.cast_unsigned(),
                flags: draw_args_flags(opaque_visible, obj.resident, obj.cullable())
                    | routes.draw_bits(obj)
                    | model_history.draw_flags(runtime_base + k, i),
            };
        });
        // Disable the unused reserve tail so vacated / never-used slots draw
        // nothing (ENABLED clear -> the cull kernel emits a no-op and never reads
        // its stale object record).
        let disabled = GpuDrawArgs {
            index_count: 0,
            index_offset: 0,
            base_vertex: 0,
            flags: 0,
        };
        let reserve_end = (runtime_base + self.state.draw.n_runtime).min(draw_args.len());
        self.write_cull_variants(slot);
        for arg in draw_args
            .get_mut(runtime_base + n_resident_runtime..reserve_end)
            .unwrap_or_default()
        {
            *arg = disabled;
        }

        // Skinned objects in the reserved tail: cullable by a padded bind-pose AABB,
        // drawn by the second `ExecuteIndirect` over the deformed buffer, so
        // `base_vertex = 0`.
        let skinned_base = self.state.draw.skinned_record_base();
        let skinned = self
            .state
            .skinned
            .draw_objects
            .iter()
            .take(self.state.draw.n_skinned);
        let skinned_records = records.get_mut(skinned_base..).unwrap_or_default();
        let skinned_args = draw_args.get_mut(skinned_base..).unwrap_or_default();
        for (k, (obj, (rec, arg))) in skinned
            .zip(skinned_records.iter_mut().zip(skinned_args.iter_mut()))
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            *rec = pack_skinned_record(obj, albedo, normal);
            let d = lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            *arg = GpuDrawArgs {
                index_count: u32::saturating_from(index_count),
                index_offset: u32::saturating_from(index_offset),
                base_vertex: 0,
                flags: draw_args_flags(obj.visible, true, true)
                    | model_history.skinned_flags(skinned_base + k, k),
            };
        }

        // Patch each instance's active LOD slice, for clusters that declare alternates.
        if self.instanced.any_lod {
            let instance_base = self.state.draw.n_objects;
            lod::for_each_instance_lod(
                &self.instanced.clusters,
                cam_pos,
                |record, index_offset, index_count| {
                    if let Some(arg) = draw_args.get_mut(instance_base + record) {
                        *arg = GpuDrawArgs {
                            index_count: u32::saturating_from(index_count),
                            index_offset: u32::saturating_from(index_offset),
                            base_vertex: 0,
                            flags: draw_args_flags(true, true, true),
                        };
                    }
                },
            );
        }
        packing
    }

    // Dispatch the main cull: test every object against the frustum, distance and
    // (when valid) Hi-Z, and write the survivors' commands. The graph drives the
    // indirect buffer's transitions. The caller already packed the records.
    pub(in crate::directx) fn encode_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let Some(kernels) = self.cull.cull_kernels.as_ref() else {
            return;
        };
        let (cull_pso, cull_root) = (&kernels.pso, &kernels.root_sig);
        let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        let draw_args_gva = com::gpu_va(&self.cull.draw_args_buffer_resources[frame_idx]);
        // Per-object cull-status buffer (u1): always allocated alongside the
        // indirect buffer, always written, read by the late pass under two-pass
        // occlusion (ignored under single-pass). Resting state is
        // `UNORDERED_ACCESS`, so it binds as a root UAV with no transition.
        let cull_status_gva = com::gpu_va(&self.cull.cull_status_buffers[frame_idx]);

        // Hi-Z is off until a pyramid exists (`hiz_valid`) or without Hi-Z resources.
        let (hiz_size, hiz_mip_count, hiz_srv, hiz_enabled) = match self.cull.hiz.as_ref() {
            Some(h) if self.state.planner.temporal().occlusion().valid() => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
                1u32,
            ),
            Some(h) => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
                0u32,
            ),
            None => ([1.0, 1.0], 1, None, 0u32),
        };
        let mut cull_params = CullParams {
            view: CullView {
                planes: [[0.0; 4]; 6],
                object_count: u32::saturating_from(self.state.draw.record_count()),
                bucket_count: u32::saturating_from(self.shader_bucket_count()),
                skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
                exclude_first: 0,
                exclude_end: 0,
                cam_pos,
            },
            hiz: CullHiz {
                prev_view_proj: self.state.planner.temporal().occlusion().prev_view_proj(),
                hiz_size,
                hiz_mip_count,
                hiz_enabled,
            },
        };
        cull_params.view.planes = frustum.packed_planes();

        // SAFETY: the command list is recording, and the cull root signature is owned by
        // `self.cull` for the context's lifetime.
        unsafe { cmd.SetComputeRootSignature(cull_root) };
        // SAFETY: the cull PSO is owned by `self.cull` and matches the root signature above.
        unsafe { cmd.SetPipelineState(cull_pso) };
        // Bind the SRV heap for the Hi-Z table, which must name a live descriptor even
        // when Hi-Z is off.
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: parameter 0 of the cull root signature holds `CullParams`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, &cull_params) };
        // SAFETY: `object_gva` addresses this frame's live object buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(1, object_gva) };
        // SAFETY: `draw_args_gva` addresses this frame's live draw-args buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(2, draw_args_gva) };
        if let Some(srv) = hiz_srv {
            // SAFETY: the Hi-Z SRV slot lives in the heap set above.
            unsafe { cmd.set_compute_srv_table(3, srv) };
        }
        // SAFETY: the slot's group counts rest in UNORDERED_ACCESS, so they bind with no
        // transition.
        unsafe { cmd.SetComputeRootUnorderedAccessView(4, self.group_counts_gva(frame_idx)) };
        // SAFETY: the cull-status buffer rests in UNORDERED_ACCESS, so it binds with no transition.
        unsafe { cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva) };
        self.bind_cull_variants(cmd, frame_idx);
        // One thread per build-time object.
        let [x, y, z] = CULL_GROUP.groups_1d(u32::saturating_from(self.state.draw.record_count()));
        // SAFETY: the cull pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(x, y, z) };
        if let Some(pass) = self.compact_pass(
            frame_idx,
            self.main_lists(indirect),
            self.shader_bucket_count(),
            CullStatus::DRAWN,
        ) {
            self.encode_compaction(cmd, pass);
        }
    }

    // The main view's lists in `buffer`, a slot's early or late indirect
    // buffer.
    pub(in crate::directx) fn main_lists<'a>(&self, buffer: &'a ID3D12Resource) -> DrawLists<'a> {
        DrawLists {
            buffer,
            base: 0,
            commands: self.cull.indirect_commands,
        }
    }

    // How the main view's lists sit in its indirect buffers.
    pub(in crate::directx) fn stream_layout(&self) -> StreamLayout<'_> {
        StreamLayout {
            regions: &self.cull.variants.regions,
            bucket_count: u32::saturating_from(self.shader_bucket_count()),
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
            records: u32::saturating_from(self.state.draw.record_count()),
        }
    }

    // GPU address of slot `slot`'s group-count scratch.
    fn group_counts_gva(&self, slot: usize) -> u64 {
        self.cull
            .group_count_buffers
            .get(slot)
            .map_or(0, com::gpu_va)
    }

    // Slot `slot`'s packing into `lists`, routing between `buckets` and
    // keeping the records with outcome `drawn`.
    fn compact_pass<'a>(
        &'a self,
        slot: usize,
        lists: DrawLists<'a>,
        buckets: usize,
        drawn: u32,
    ) -> Option<CompactPass<'a>> {
        Some(CompactPass {
            variants: com::gpu_va(self.cull.variant_buffers.get(slot)?),
            draw_args: com::gpu_va(self.cull.draw_args_buffer_resources.get(slot)?),
            group_counts: UavSpan {
                buffer: self.cull.group_count_buffers.get(slot)?,
                offset: 0,
            },
            status: UavSpan {
                buffer: self.cull.cull_status_buffers.get(slot)?,
                offset: 0,
            },
            lists,
            params: GpuCompactParams {
                object_count: u32::saturating_from(self.state.draw.record_count()),
                bucket_count: u32::saturating_from(buckets),
                skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
                drawn,
                ..Default::default()
            },
        })
    }

    // Cull one probe face into the reserved capture slot, with Hi-Z off and the
    // slot's records built once at capture start. Skinned records are written but
    // never executed, since the bake draws only the static + instance prefix.
    pub(in crate::directx) fn encode_probe_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        slot: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let Some(kernels) = self.cull.cull_kernels.as_ref() else {
            return;
        };
        let (cull_pso, cull_root) = (&kernels.pso, &kernels.root_sig);
        let indirect = &self.cull.indirect_cmd_buffers[slot];
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[slot]);
        let draw_args_gva = com::gpu_va(&self.cull.draw_args_buffer_resources[slot]);
        let cull_status_gva = com::gpu_va(&self.cull.cull_status_buffers[slot]);

        // A Hi-Z SRV is still bound (the root signature's descriptor table at [3]
        // must point at a live descriptor) but `hiz_enabled = 0` makes the kernel
        // skip the occlusion test. The probe captures whatever is in the face
        // frustum, occluded or not.
        let hiz_srv = self.cull.hiz.as_ref().map(|h| h.srv_gpu);
        let mut cull_params = CullParams {
            view: CullView {
                planes: [[0.0; 4]; 6],
                object_count: u32::saturating_from(self.state.draw.record_count()),
                // The bake renders one command stream under the default bindless
                // pipeline, so every record is routed into region 0: a bucketed draw
                // is captured into the probe cube with default shading.
                bucket_count: 1,
                skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
                exclude_first: 0,
                exclude_end: 0,
                cam_pos,
            },
            hiz: CullHiz {
                prev_view_proj: self.state.planner.temporal().occlusion().prev_view_proj(),
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
            },
        };
        cull_params.view.planes = frustum.packed_planes();

        let to_uav = transition_barrier(
            indirect,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the command list is recording, and the slot's indirect buffer rests in
        // INDIRECT_ARGUMENT between uses.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };
        // SAFETY: the cull root signature is owned by `self.cull` for the context's lifetime.
        unsafe { cmd.SetComputeRootSignature(cull_root) };
        // SAFETY: the cull PSO is owned by `self.cull` and matches the root signature above.
        unsafe { cmd.SetPipelineState(cull_pso) };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: parameter 0 of the cull root signature holds `CullParams`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, &cull_params) };
        // SAFETY: `object_gva` addresses the slot's live object buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(1, object_gva) };
        // SAFETY: `draw_args_gva` addresses the slot's live draw-args buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(2, draw_args_gva) };
        if let Some(srv) = hiz_srv {
            // SAFETY: the Hi-Z SRV slot lives in the heap set above and stays a live descriptor.
            unsafe { cmd.set_compute_srv_table(3, srv) };
        }
        // SAFETY: the slot's group counts rest in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(4, self.group_counts_gva(slot)) };
        // SAFETY: the slot's cull-status buffer is live and rests in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva) };
        self.bind_cull_variants(cmd, slot);
        let [x, y, z] = CULL_GROUP.groups_1d(u32::saturating_from(self.state.draw.record_count()));
        // SAFETY: the cull pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(x, y, z) };
        // The face draws one stream, packed into the indirect buffer moved to
        // UNORDERED_ACCESS above.
        if let Some(pass) = self.compact_pass(slot, self.main_lists(indirect), 1, CullStatus::DRAWN)
        {
            self.encode_compaction(cmd, pass);
        }
        let to_indirect = transition_barrier(
            indirect,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
        // SAFETY: the indirect buffer was moved to UNORDERED_ACCESS by the first barrier.
        unsafe { cmd.ResourceBarrier(&[to_indirect]) };
    }

    // Cull every record against each re-rendered cascade's light frustum, with no
    // Hi-Z and no distance cull, into that cascade's region. Skipped cascades keep
    // their region. A no-op without the GPU-driven shadow path.
    pub(in crate::directx) fn encode_shadow_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        render_mask: u32,
    ) {
        use concinnity_engine::gfx::render_types::NUM_SHADOW_CASCADES;
        let (Some(pso), Some(views)) = (
            self.cull.cull_pso_shadow.as_ref(),
            self.cull.shadow_views.get(frame_idx),
        ) else {
            return;
        };
        let mut regions = [RegionCull::EMPTY; NUM_SHADOW_CASCADES];
        let mut kept = 0;
        for c in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << c) == 0 {
                continue;
            }
            regions[kept] = RegionCull {
                region: c,
                frustum: self.state.planner.shadow.cascade_cull(c),
                ..RegionCull::EMPTY
            };
            kept += 1;
        }
        self.encode_region_culls(
            cmd,
            frame_idx,
            self.shadow_cull_pass(pso, views),
            &regions[..kept],
        );
    }

    // Cull each refreshed spot slice's casters against the spot frustum into its
    // region; casters keep the frame's camera-keyed LOD.
    pub(in crate::directx) fn encode_spot_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let (Some(pso), Some(views)) = (
            self.cull.cull_pso_shadow.as_ref(),
            self.cull.spot_views.get(frame_idx),
        ) else {
            return;
        };
        let mut regions = [RegionCull::EMPTY; MAX_SHADOWED_SPOTS];
        let mut kept = 0;
        for slice in self.spot_shadow.refreshed_slices() {
            regions[kept] = RegionCull {
                region: slice as usize,
                frustum: self.spot_shadow.frusta[slice as usize],
                ..RegionCull::EMPTY
            };
            kept += 1;
        }
        self.encode_region_culls(
            cmd,
            frame_idx,
            self.shadow_cull_pass(pso, views),
            &regions[..kept],
        );
    }

    // The far view's cull against the projection this frame redraws it
    // through, keeping only the casters large enough for its texels, into the
    // one region of this frame's far views.
    pub(in crate::directx) fn encode_far_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let (Some(pso), Some(views)) = (
            self.cull.cull_pso_shadow.as_ref(),
            self.cull.far_views.get(frame_idx),
        ) else {
            return;
        };
        let far = &self.state.planner.shadow.uniforms.far;
        let region = RegionCull {
            frustum: Frustum::from_shadow(far.light_vp),
            min_caster_radius: far_caster_radius(far),
            ..RegionCull::EMPTY
        };
        self.encode_region_culls(cmd, frame_idx, self.shadow_cull_pass(pso, views), &[region]);
    }

    // A shadow pass's run of region culls, each view packing its kept draws
    // into its own region of `views`.
    fn shadow_cull_pass<'a>(
        &self,
        pso: &'a ID3D12PipelineState,
        views: &'a ShadowCullViews,
    ) -> RegionCullPass<'a> {
        use crate::directx::compact::{group_count_bytes, indirect_buffer_bytes};
        let n_cull = self.cull.bucket_stride;
        RegionCullPass {
            pso,
            kernel: RegionKernel::Shadow,
            indirect: &views.lists,
            status_gva: com::gpu_va(&views.status),
            region_stride: n_cull,
            packing: Some(RegionPacking {
                group_counts: &views.group_counts,
                group_stride: group_count_bytes(n_cull),
                status: &views.status,
                status_stride: shadow_status_bytes(n_cull),
                lists_stride: indirect_buffer_bytes(n_cull),
            }),
        }
    }

    // View `region`'s lists in `views`, for the shadow draws.
    pub(in crate::directx) fn shadow_lists<'a>(
        &self,
        views: &'a ShadowCullViews,
        region: usize,
    ) -> DrawLists<'a> {
        let n_cull = self.cull.bucket_stride;
        DrawLists {
            buffer: &views.lists,
            base: region as u64 * crate::directx::compact::indirect_buffer_bytes(n_cull),
            commands: n_cull,
        }
    }

    // How a shadow view's lists sit in its region: one stream and the skinned
    // tail, over the records the view's cull covers.
    pub(in crate::directx) fn shadow_layout(&self) -> StreamLayout<'_> {
        StreamLayout {
            regions: &self.cull.variants.regions,
            bucket_count: 1,
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
            records: culled_records(self.state.draw.record_count(), self.cull.bucket_stride),
        }
    }

    // Cull the frame's camera-independent records against each kept plane's
    // reflected frustum (oblique clip included), Hi-Z off, into its region.
    pub(in crate::directx) fn encode_planar_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        planes: &[RegionCull],
        indirect: &ID3D12Resource,
        // Per-plane capacity: the fixed build-time record capacity the reader uses, not
        // the live record count, which can be smaller.
        region_count: usize,
        packing: RegionPacking<'_>,
    ) {
        let Some(pso) = self.cull.cull_kernels.as_ref().map(|k| &k.pso) else {
            return;
        };
        self.encode_region_culls(
            cmd,
            frame_idx,
            RegionCullPass {
                pso,
                kernel: RegionKernel::Camera,
                indirect,
                status_gva: com::gpu_va(packing.status),
                region_stride: region_count,
                packing: Some(packing),
            },
            planes,
        );
    }

    // One cull per region, each into `region * region_stride` of `pass.indirect`,
    // every record into region 0 (one stream, never bucketed). The buffer flips to
    // UAV once for all dispatches.
    fn encode_region_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        pass: RegionCullPass<'_>,
        regions: &[RegionCull],
    ) {
        let Some(cull_root) = self.cull.cull_kernels.as_ref().map(|k| &k.root_sig) else {
            return;
        };
        let n_cull = culled_records(self.state.draw.record_count(), pass.region_stride);
        if n_cull == 0 || regions.is_empty() {
            return;
        }

        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        let draw_args_gva = com::gpu_va(&self.cull.draw_args_buffer_resources[frame_idx]);
        let base_gva = com::gpu_va(pass.indirect);

        // Hi-Z disabled (`hiz_enabled = 0`): the kernel never samples the pyramid,
        // but the descriptor table at root [3] must still point at a live descriptor.
        let (hiz_size, hiz_mip_count, hiz_srv) = match self.cull.hiz.as_ref() {
            Some(h) => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
            ),
            None => ([1.0, 1.0], 1, None),
        };

        let to_uav = transition_barrier(
            pass.indirect,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the command list is recording, and the pass's indirect buffer rests in
        // INDIRECT_ARGUMENT between uses.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };
        // The cull's root arguments, bound again after each region's packing,
        // which runs under its own root signature.
        let bind = || {
            // SAFETY: the cull root signature is owned by `self.cull` for the context's lifetime.
            unsafe { cmd.SetComputeRootSignature(cull_root) };
            // SAFETY: the pass's PSO is a live cull kernel built against the root signature above.
            unsafe { cmd.SetPipelineState(pass.pso) };
            // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
            unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
            // SAFETY: `object_gva` addresses this frame's live object buffer.
            unsafe { cmd.SetComputeRootShaderResourceView(1, object_gva) };
            // SAFETY: `draw_args_gva` addresses this frame's live draw-args buffer.
            unsafe { cmd.SetComputeRootShaderResourceView(2, draw_args_gva) };
            if let Some(srv) = hiz_srv {
                // SAFETY: the Hi-Z SRV slot is a live descriptor in the heap set above.
                unsafe { cmd.set_compute_srv_table(3, srv) };
            }
            // SAFETY: `pass.status_gva` addresses the pass's live cull-status buffer.
            unsafe { cmd.SetComputeRootUnorderedAccessView(5, pass.status_gva) };
            self.bind_cull_variants(cmd, frame_idx);
        };
        bind();

        for (i, r) in regions.iter().enumerate() {
            if i > 0 && pass.packing.is_some() {
                bind();
            }
            let skinned_base = u32::saturating_from(self.state.draw.skinned_record_base());
            match pass.kernel {
                RegionKernel::Camera => {
                    let cull_params = CullParams {
                        view: CullView {
                            planes: r.frustum.packed_planes(),
                            object_count: n_cull,
                            bucket_count: 1,
                            skinned_base,
                            exclude_first: r.exclude[0],
                            exclude_end: r.exclude[1],
                            cam_pos: r.eye,
                        },
                        hiz: CullHiz {
                            // Unused with Hi-Z disabled (the reprojection is never taken).
                            prev_view_proj: [[0.0; 4]; 4],
                            hiz_size,
                            hiz_mip_count,
                            hiz_enabled: 0,
                        },
                    };
                    // SAFETY: parameter 0 of the cull root signature holds `CullParams`-sized
                    // root constants.
                    unsafe { cmd.set_compute_root_constants(0, &cull_params) };
                }
                RegionKernel::Shadow => {
                    let params = ShadowCullParams::new(
                        &r.frustum,
                        n_cull,
                        skinned_base,
                        r.min_caster_radius,
                    );
                    // SAFETY: parameter 0 of the cull root signature holds
                    // `CullParams`-sized root constants, which the smaller shadow
                    // block fits inside.
                    unsafe { cmd.set_compute_root_constants(0, &params) };
                }
            }
            let Some(packing) = pass.packing else {
                // Root UAV GPU addresses only need element alignment (the stride is
                // a multiple of 4), like the instanced path's per-bucket root-SRV bumps.
                let region_gva = base_gva
                    + (r.region * pass.region_stride * INDIRECT_COMMAND_STRIDE as usize) as u64;
                // SAFETY: the cull covers no more records than the region stride, so the region's
                // commands stay inside the indirect buffer opened for unordered access above.
                unsafe { cmd.SetComputeRootUnorderedAccessView(4, region_gva) };
                let [x, y, z] = CULL_GROUP.groups_1d(n_cull);
                // SAFETY: the cull pipeline and every root argument it reads are bound above.
                unsafe { cmd.Dispatch(x, y, z) };
                continue;
            };
            let region = r.region as u64;
            let group_counts = UavSpan {
                buffer: packing.group_counts,
                offset: region * packing.group_stride,
            };
            let status = UavSpan {
                buffer: packing.status,
                offset: region * packing.status_stride,
            };
            let counts_gva = com::gpu_va(group_counts.buffer) + group_counts.offset;
            // SAFETY: the region's group counts lie inside the pass's scratch, which rests in
            // UNORDERED_ACCESS.
            unsafe { cmd.SetComputeRootUnorderedAccessView(4, counts_gva) };
            let status_gva = com::gpu_va(status.buffer) + status.offset;
            // SAFETY: as above, for the region's status.
            unsafe { cmd.SetComputeRootUnorderedAccessView(5, status_gva) };
            let [x, y, z] = CULL_GROUP.groups_1d(n_cull);
            // SAFETY: the cull pipeline and every root argument it reads are bound above.
            unsafe { cmd.Dispatch(x, y, z) };
            let Some(variants) = self.cull.variant_buffers.get(frame_idx) else {
                continue;
            };
            self.encode_compaction(
                cmd,
                CompactPass {
                    variants: com::gpu_va(variants),
                    draw_args: draw_args_gva,
                    group_counts,
                    status,
                    lists: DrawLists {
                        buffer: pass.indirect,
                        base: region * packing.lists_stride,
                        commands: pass.region_stride,
                    },
                    params: GpuCompactParams {
                        object_count: n_cull,
                        bucket_count: 1,
                        skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
                        drawn: CullStatus::DRAWN,
                        ..Default::default()
                    }
                    .with_capture_tiles(r.capture_tiles),
                },
            );
        }

        let to_indirect = transition_barrier(
            pass.indirect,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
        // SAFETY: the indirect buffer was moved to UNORDERED_ACCESS by the first barrier.
        unsafe { cmd.ResourceBarrier(&[to_indirect]) };
    }
    // The late cull for two-pass occlusion, after `HizEarly`: re-test the early
    // pass's Hi-Z candidates against the fresh pyramid through this frame's
    // un-jittered VP, and write the visible ones for `MainLate`. The graph drives
    // the transitions.
    pub(in crate::directx) fn encode_cull_late(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frustum: &Frustum,
        cur_vp: [[f32; 4]; 4],
    ) {
        let (Some(kernels), Some(hiz), Some(indirect)) = (
            self.cull.cull_kernels.as_ref(),
            self.cull.hiz.as_ref(),
            self.cull.indirect_cmd_buffers_late.get(frame_idx),
        ) else {
            return;
        };
        let Some(cull_pso_late) = kernels.pso_late.as_ref() else {
            return;
        };
        let cull_root = &kernels.root_sig;
        if self.state.draw.record_count() == 0 {
            return;
        }

        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        let draw_args_gva = com::gpu_va(&self.cull.draw_args_buffer_resources[frame_idx]);
        let cull_status_gva = com::gpu_va(&self.cull.cull_status_buffers[frame_idx]);

        // `hiz_enabled = 1`: `HizEarly` always precedes this. The frustum planes are
        // unused here but share the cbuffer layout.
        let mut cull_params = CullParams {
            view: CullView {
                planes: [[0.0; 4]; 6],
                object_count: u32::saturating_from(self.state.draw.record_count()),
                bucket_count: u32::saturating_from(self.shader_bucket_count()),
                skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
                exclude_first: 0,
                exclude_end: 0,
                cam_pos: [0.0; 3],
            },
            hiz: CullHiz {
                prev_view_proj: cur_vp,
                hiz_size: [hiz.width as f32, hiz.height as f32],
                hiz_mip_count: hiz.mip_count,
                hiz_enabled: 1,
            },
        };
        cull_params.view.planes = frustum.packed_planes();

        // SAFETY: the command list is recording, and the cull root signature is owned by
        // `self.cull` for the context's lifetime.
        unsafe { cmd.SetComputeRootSignature(cull_root) };
        // SAFETY: the late PSO is owned by `self.cull` and built against the root signature
        // above.
        unsafe { cmd.SetPipelineState(cull_pso_late) };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: parameter 0 of the cull root signature holds `CullParams`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, &cull_params) };
        // SAFETY: `object_gva` addresses this frame's live object buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(1, object_gva) };
        // SAFETY: `draw_args_gva` addresses this frame's live draw-args buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(2, draw_args_gva) };
        // The rebuilt Hi-Z pyramid (same all-mips SRV phase 1 sampled; the
        // HizEarly node rewrote the texels in place).
        // SAFETY: the Hi-Z SRV slot is a live descriptor in the heap set above.
        unsafe { cmd.set_compute_srv_table(3, hiz.srv_gpu) };
        // SAFETY: the early pass's packing has read this slot's group counts (the graph orders Main and
        // its pre-pass between), and they rest in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(4, self.group_counts_gva(frame_idx)) };
        // SAFETY: the early pass wrote this frame's cull-status buffer, which rests in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva) };
        self.bind_cull_variants(cmd, frame_idx);
        let [x, y, z] = CULL_GROUP.groups_1d(u32::saturating_from(self.state.draw.record_count()));
        // SAFETY: the late pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(x, y, z) };
        if let Some(pass) = self.compact_pass(
            frame_idx,
            self.main_lists(indirect),
            self.shader_bucket_count(),
            CullStatus::REDRAW,
        ) {
            self.encode_compaction(cmd, pass);
        }
    }

    // True when two-pass Hi-Z occlusion runs this frame: requested, built, and on
    // the bindless path with geometry. The graph's HizEarly / CullLate / MainLate
    // chain and every encoder gate on exactly this.
    pub(in crate::directx) fn two_pass_occlusion_active(&self) -> bool {
        self.two_pass_occlusion_built() && self.state.draw.record_count() > 0
    }

    // Whether launch built two-pass occlusion: requested, with the phase-two
    // cull, the Hi-Z pyramid and the bindless main pass.
    pub(in crate::directx) fn two_pass_occlusion_built(&self) -> bool {
        self.cull.occlusion_two_pass
            && self
                .cull
                .cull_kernels
                .as_ref()
                .is_some_and(|k| k.pso_late.is_some())
            && self.cull.hiz.is_some()
            && self.cull.main_bindless_pso.is_some()
    }
}

// The first `len` records of a mapped upload buffer as `T`s, for an in-place
// pack. `None` when null or misaligned.
//
// # Safety
//
// `ptr` must start a live mapping of at least `len` records, and while the
// slice lives nothing else may touch those bytes.
unsafe fn mapped_records<'a, T: bytemuck::AnyBitPattern + bytemuck::NoUninit>(
    ptr: *mut u8,
    len: usize,
) -> Option<&'a mut [T]> {
    let base = ptr.cast::<T>();
    if base.is_null() || !base.is_aligned() {
        return None;
    }
    // SAFETY: the caller guarantees `len` records of live mapping from `base`
    // and exclusive access to them; the check above proved it aligned, and any
    // bit pattern is a valid `T`.
    Some(unsafe { std::slice::from_raw_parts_mut(base, len) })
}

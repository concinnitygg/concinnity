//! Hardware ray-traced reflection pass for the D3D12 backend, screen space
//! first. A classify kernel walks each glossy pixel's reflection ray through
//! SSGI's depth pyramid, writes what the screen and the probe settle into the
//! output, and lists the pixels still owed a hardware ray; a trace kernel
//! dispatched through `ExecuteIndirect` over that list traces them against the
//! scene's DXR top-level acceleration structure ([`crate::directx::raytrace`])
//! with `RayQuery`, shading the hit (sun + IBL split-sum, optionally textured)
//! or the probe / IBL prefilter cube on a miss, and adds their share. The
//! reflection composite then blends the output over the scene.
//!
//! It occupies the `SsrResolve` slot in the frame graph (reads `hdr_resolve`,
//! writes its own output target) and is mutually exclusive with SSR resolve.
//! Like SSGI it relies on the SSR pre-pass G-buffer, so the pre-pass is forced on
//! whenever RT reflections are enabled. The shader is the shared
//! `shaders/rt_reflections.hlsl`; the trace compiles to a shader-model 6.5 DXIL
//! container (the floor for the inline ray query) through `builtin_shaders`; the
//! Metal and Vulkan hosts bind the same source at their own slots.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::RtParams;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::rt_reflections::{
    RAY_ARGS_RESET, RtReflectionSettings, classify_groups, ray_list_len,
};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::post::descriptors::{PostDescriptors, PostSrvRun};
use crate::directx::post::post_device::PostTarget;
use crate::directx::pso::compute_pso;
use crate::directx::root_sig::{Range, RootSig, SamplerState, Visibility};
use crate::directx::texture::{
    HDR_FORMAT, create_storage_target, create_uav_buffer, transition_barrier, uav_barrier,
    write_format_srv, write_format_uav,
};
use crate::directx::{builtin_shaders, com};

// Size of the RT-reflection uniform block.
const RT_PARAMS_UBO_SIZE: u64 = size_of::<RtParams>() as u64;

// Bytes of the ray arguments: three group counts and the listed pixel count.
const RAY_ARGS_BYTES: u64 = size_of::<[u32; 4]>() as u64;

// Where the ray arguments rest between frames, ready for the next reset.
const RAY_ARGS_REST: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_DEST;
// What the trace reads them through: the indirect dispatch and its shader.
const RAY_ARGS_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);
// The SSGI pyramid's resting state, and the one the classify reads it in.
const PYRAMID_REST: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
const PYRAMID_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;

// Root parameters past the probe and canopy bindings.
const ROOT_OUTPUT: u32 = 19;
const ROOT_PYRAMID: u32 = 20;
const ROOT_RAY_LIST_UAV: u32 = 21;
const ROOT_RAY_ARGS_UAV: u32 = 22;
const ROOT_RAY_LIST_SRV: u32 = 23;
const ROOT_RAY_ARGS_SRV: u32 = 24;

// Shader compilation

struct RtShaders {
    classify: Vec<u8>,
    flat: Vec<u8>,
    textured: Vec<u8>,
}

// Compile the classify and both trace kernels from the single source (the
// trace at SM 6.5, for the inline ray query). Returns an `Err` (which the
// caller turns into an SSR fallback) when the toolchain is unavailable or a
// kernel fails to compile.
fn compile_rt_shaders(hot_reload: bool) -> RenderResult<RtShaders> {
    Ok(RtShaders {
        classify: builtin_shaders::RT_REFLECTIONS_CLASSIFY.compile(hot_reload)?,
        flat: builtin_shaders::RT_REFLECTIONS_TRACE.compile(hot_reload)?,
        textured: builtin_shaders::RT_REFLECTIONS_TRACE_TEXTURED.compile(hot_reload)?,
    })
}

// Root signature
//
// One compute root signature both kernels share: the RtParams root CBV; the
// TLAS, vertex, index and geometry table as root SRVs (raw or structured
// buffers bound by GPU virtual address, which inline ray tracing supports for
// the acceleration structure too); descriptor tables for the scene, the
// G-buffer normal+depth, the roughness, the prefilter cube and the unbounded
// bindless pool; root SRVs for the deformed skinned vertices and indices, for
// skinned hits. The miss fallback's reflection probes follow: the cube array
// table, the ProbeSet CBV, the records, and the main camera's cluster grid
// binning them (its params CBV and per-cluster lists). The textured trace's
// grass-covered hits shade from the canopy CBV and its view tables. Then the
// output's UAV table, SSGI's pyramid table, and the ray list and arguments:
// root UAVs the classify writes, root SRVs the trace reads. Four static
// samplers: the screen sources', the prefilter cube's, the pool's and the probe
// cube array's.
fn create_rt_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    use Visibility::All;
    use concinnity_engine::render::shader::programs::abi::rt_reflections::dx as rt;
    RootSig::new()
        .cbv(rt::PARAMS, All) // [0]
        .srv(rt::SCENE_TLAS, All) // [1]
        .srv(rt::VERTICES, All) // [2] raw
        .srv(rt::INDICES, All) // [3] raw
        .srv(rt::GEOMETRY, All) // [4] structured
        .srv_table(rt::SCENE, 1, All) // [5]
        .srv_table(rt::GBUFFER, 1, All) // [6] normal+depth
        .srv_table(rt::ROUGHNESS, 1, All) // [7]
        .srv_table(rt::PREFILTER, 1, All) // [8]
        .table(&[Range::bindless_srv(rt::TEX_POOL)], All) // [9]
        .srv(rt::SKINNED_VERTICES, All) // [10] raw
        .srv(rt::SKINNED_INDICES, All) // [11] raw
        // [12] the reflection-probe cube array, clear of the trace's own SRVs;
        // the miss fallback box-projects it when ProbeSet.count > 0.
        .srv_table(rt::PROBE_CUBES, 1, All)
        .cbv(rt::PROBE_SET, All) // [13]
        .srv(rt::PROBE_RECORDS, All) // [14]
        .cbv(rt::CLUSTER, All) // [15]
        .srv(rt::CLUSTER_LIST, All) // [16]
        .cbv(rt::CANOPY, All) // [17]
        .srv(rt::CANOPY_TABLES, All) // [18]
        .uav_table(rt::OUTPUT, 1, All) // [19]
        .srv_table(rt::PYRAMID, 1, All) // [20]
        .uav(rt::RAY_LIST_OUT, All) // [21]
        .uav(rt::RAY_ARGS_OUT, All) // [22]
        .srv(rt::RAY_LIST, All) // [23]
        .srv(rt::RAY_ARGS, All) // [24]
        .static_sampler(SamplerState::LinearClamp, rt::SCREEN_SAMPLER, All)
        .static_sampler(SamplerState::LinearClamp, rt::CUBE_SAMPLER, All)
        .static_sampler(SamplerState::LinearWrap, rt::POOL_SAMPLER, All)
        // The cube mip-linear clamp for the reflection-probe cube array, matching
        // the prefilter sampler. The array rides split from its sampler here
        // because D3D12 binds a shader sampler array only through a descriptor
        // table.
        .static_sampler(SamplerState::LinearClamp, rt::PROBE_CUBE_SAMPLER, All)
        .build(device, "rt reflections root sig")
}

// The classify and trace PSOs over the shared root signature.
fn create_rt_psos(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    root_sig: &ID3D12RootSignature,
    shaders: &RtShaders,
) -> RenderResult<RebuiltRtPipelines> {
    let pso =
        |cs: &[u8], label: &str| dump_on_err(info_queue, compute_pso(device, root_sig, cs, label));
    Ok(RebuiltRtPipelines {
        classify: pso(&shaders.classify, "rt reflections classify")?,
        flat: pso(&shaders.flat, "rt reflections trace")?,
        textured: pso(&shaders.textured, "rt reflections trace textured")?,
    })
}

// The command signature the trace dispatches through: one `Dispatch` per
// record, whose three group counts lead the ray arguments.
fn create_dispatch_signature(device: &ID3D12Device) -> RenderResult<ID3D12CommandSignature> {
    let arg = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: u32::saturating_from(RAY_ARGS_BYTES),
        NumArgumentDescs: 1,
        pArgumentDescs: &raw const arg,
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    // SAFETY: the create descriptor and the argument it points at are live for the call, and
    // the new COM object lands in a binding that owns it. A dispatch-only signature names no
    // root parameter, so it takes no root signature.
    unsafe { device.CreateCommandSignature(&raw const desc, None, &raw mut sig) }
        .map_err(|e| map_hresult(e.code(), "create rt dispatch signature"))?;
    sig.ok_or_else(|| RenderError::Other("create rt dispatch signature: returned None".into()))
}

// Resources

// The output and ray list a new trace resolution replaced, held until no frame
// in flight reads them.
pub(in crate::directx) struct RetiredRtOutput {
    _targets: RtTargets,
}

// What the trace resolution sizes: the output, its SRV (slot 0) and UAV (slot
// 1) from the post block, and the ray list with one entry per output texel.
struct RtTargets {
    output: ID3D12Resource,
    _descriptors: PostSrvRun,
    srv: SrvSlot,
    uav: SrvSlot,
    ray_list: ID3D12Resource,
}

impl RtTargets {
    fn new(
        device: &ID3D12Device,
        post: &PostDescriptors,
        extent: (u32, u32),
    ) -> RenderResult<Self> {
        let output = create_storage_target(device, extent.0, extent.1, HDR_FORMAT)?;
        let descriptors = post.allocate_srv_run(2)?;
        let (Some(srv_cpu), Some(uav_cpu), Some(srv), Some(uav)) = (
            descriptors.cpu(0),
            descriptors.cpu(1),
            descriptors.gpu(0),
            descriptors.gpu(1),
        ) else {
            return Err(RenderError::Other(
                "rt reflections: descriptor run too short".into(),
            ));
        };
        write_format_srv(device, &output, srv_cpu, HDR_FORMAT);
        write_format_uav(device, &output, uav_cpu, HDR_FORMAT);
        // Created in COMMON, which a buffer's first UAV use promotes from; the
        // trace's read and the transition back keep it in UNORDERED_ACCESS
        // between frames.
        let ray_list = create_uav_buffer(device, ray_list_len(extent.0, extent.1) * 4)?;
        Ok(Self {
            output,
            _descriptors: descriptors,
            srv,
            uav,
            ray_list,
        })
    }
}

// Hardware-ray-traced-reflection resources held by `DxContext` when the world's
// `Reflections` enables `ray_traced` AND the GPU supports the
// DXR tier AND the DXC compile + acceleration-structure build succeed; otherwise
// the context leaves this `None` and the graph falls back to `SsrResolve`.
pub(in crate::directx) struct RtReflectionsResources {
    // Resolved authored tunables; turned into a per-frame `RtParams` push.
    pub(in crate::directx) settings: RtReflectionSettings,

    // Reflection output (reflected radiance + composite weight at the trace
    // resolution, which the reflection composite upsamples over the scene) and
    // the ray list. The output owns its own slot rather than reusing the
    // optional SSR resolve output, because RT can be authored with SSR resolve
    // off.
    targets: RtTargets,
    // The trace resolution: render resolution reduced by `settings.divisor`.
    extent: (u32, u32),
    // The trace's indirect arguments, and the upload copy every frame's
    // classify starts them from.
    ray_args: ID3D12Resource,
    ray_args_reset: PooledBuffer,

    // Per-frame `RtParams` UBO, persistently mapped.
    params_ubo_resources: Vec<PooledBuffer>,
    params_ubo_ptrs: Vec<*mut u8>,

    // One root signature shared by every PSO; the classify, and the flat-tint
    // and textured (bindless) traces.
    root_sig: ID3D12RootSignature,
    dispatch_signature: ID3D12CommandSignature,
    classify_pso: ID3D12PipelineState,
    flat_pso: ID3D12PipelineState,
    textured_pso: ID3D12PipelineState,
}

// Device + render extent for building the RT-reflection resources; the output
// target is that extent reduced to the trace resolution.
#[derive(Clone, Copy)]
pub(in crate::directx) struct RtBuildContext<'a> {
    pub alloc: &'a DeviceAllocator,
    pub width: u32,
    pub height: u32,
}

// Optional debug info queue plus the shader hot-reload toggle for the build.
#[derive(Clone, Copy)]
pub(in crate::directx) struct RtBuildInit<'a> {
    pub info_queue: Option<&'a ID3D12InfoQueue>,
    pub hot_reload: bool,
}

// A persistently mapped upload buffer of `size` bytes.
fn mapped_upload(alloc: &DeviceAllocator, size: u64) -> RenderResult<(PooledBuffer, *mut u8)> {
    let buf = alloc.alloc_buffer(
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
    // that receives the mapping.
    unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "map rt upload buffer"))?;
    Ok((buf, ptr.cast::<u8>()))
}

impl RtReflectionsResources {
    // Build the RT-reflection resources. Returns `Err` when the DXC compile
    // fails (DXC absent / shader error) so the caller can fall back to SSR; the
    // accel-structure build + DXR capability are gated separately by the caller.
    pub(in crate::directx) fn new(
        ctx: RtBuildContext<'_>,
        settings: RtReflectionSettings,
        post: &PostDescriptors,
        init: RtBuildInit<'_>,
    ) -> RenderResult<Self> {
        let RtBuildContext {
            alloc,
            width,
            height,
        } = ctx;
        let device = alloc.device();
        let RtBuildInit {
            info_queue,
            hot_reload,
        } = init;
        let extent = settings.trace_extent(width, height);
        let targets = RtTargets::new(device, post, extent)?;

        let params_size = align256(RT_PARAMS_UBO_SIZE);
        let frames = alloc.frames_in_flight();
        let mut params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let (buf, ptr) = mapped_upload(alloc, params_size)?;
            params_ubo_ptrs.push(ptr);
            params_ubo_resources.push(buf);
        }

        let (ray_args_reset, reset_ptr) = mapped_upload(alloc, RAY_ARGS_BYTES)?;
        // SAFETY: the destination is the persistent mapping of an upload buffer sized for the
        // argument block, and the source is a separate constant, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                RAY_ARGS_RESET.as_ptr().cast::<u8>(),
                reset_ptr,
                size_of_val(&RAY_ARGS_RESET),
            );
        }
        let ray_args = create_uav_buffer(device, RAY_ARGS_BYTES)?;

        let shaders = compile_rt_shaders(hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_rt_root_signature(device))?;
        let dispatch_signature = create_dispatch_signature(device)?;
        let RebuiltRtPipelines {
            classify: classify_pso,
            flat: flat_pso,
            textured: textured_pso,
        } = create_rt_psos(device, info_queue, &root_sig, &shaders)?;

        Ok(Self {
            settings,
            targets,
            extent,
            ray_args,
            ray_args_reset,
            params_ubo_resources,
            params_ubo_ptrs,
            root_sig,
            dispatch_signature,
            classify_pso,
            flat_pso,
            textured_pso,
        })
    }

    // Rebuild the output target and ray list for a new render resolution, at
    // the trace resolution `settings` names. The caller has idled the device.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        post: &PostDescriptors,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        self.extent = self.settings.trace_extent(width, height);
        self.targets = RtTargets::new(device, post, self.extent)?;
        Ok(())
    }

    // Take `settings`, rebuilding the output and ray list at its trace
    // resolution into fresh descriptors when that moved. Returns the replaced
    // targets, which a frame in flight may still read, for the caller to
    // retire.
    pub(in crate::directx) fn retrace(
        &mut self,
        device: &ID3D12Device,
        post: &PostDescriptors,
        settings: RtReflectionSettings,
        (width, height): (u32, u32),
    ) -> RenderResult<Option<RetiredRtOutput>> {
        let extent = settings.trace_extent(width, height);
        if extent == self.extent {
            self.settings = settings;
            return Ok(None);
        }
        let targets = RtTargets::new(device, post, extent)?;
        self.settings = settings;
        self.extent = extent;
        Ok(Some(RetiredRtOutput {
            _targets: std::mem::replace(&mut self.targets, targets),
        }))
    }
}

// Replacement RT PSOs returned by [`rebuild_rt_reflections_pipelines`]. The
// caller swaps them into the live resources only if every build succeeded.
pub(in crate::directx) struct RebuiltRtPipelines {
    pub classify: ID3D12PipelineState,
    pub flat: ID3D12PipelineState,
    pub textured: ID3D12PipelineState,
}

// Rebuild every RT PSO against fresh shader source, reusing the existing root
// signature. Returns the new PSOs for the caller to swap into the live
// `RtReflectionsResources`.
pub(in crate::directx) fn rebuild_rt_reflections_pipelines(
    device: &ID3D12Device,
    rt: &RtReflectionsResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> RenderResult<RebuiltRtPipelines> {
    create_rt_psos(
        device,
        info_queue,
        &rt.root_sig,
        &compile_rt_shaders(hot_reload)?,
    )
}

// Swap freshly compiled RT PSOs into the live resources after a hot-reload.
pub(in crate::directx) fn swap_rt_reflections_pipelines(
    rt: &mut RtReflectionsResources,
    rebuilt: RebuiltRtPipelines,
) {
    rt.classify_pso = rebuilt.classify;
    rt.flat_pso = rebuilt.flat;
    rt.textured_pso = rebuilt.textured;
}

// Encoder

impl DxContext {
    // Encode the RT-reflection resolve: reset the ray arguments, classify every
    // output texel, trace the listed ones, then blend the output over the scene
    // through the reflection composite. No-op when any required resource is
    // missing (the graph only schedules this pass when RT is live, so the guards
    // are defensive).
    pub(in crate::directx) fn encode_rt_reflections(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        params: Option<&RtParams>,
    ) {
        let (Some(rt), Some(accel), Some(gbuffer), Some(params)) =
            (&self.rt_reflections, &self.rt.accel, &self.gbuffer, params)
        else {
            return;
        };
        let targets = &rt.targets;
        // SSGI's pyramid, which the classify reads only while `params` names
        // its levels; the G-buffer stands in for it otherwise.
        let pyramid = self
            .ssgi
            .as_ref()
            .filter(|_| params.cam_pos[3] > 0.0)
            .and_then(|ssgi| ssgi.pyramid().ok());

        // Upload this frame's RtParams.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<RtParams>(params).cast::<u8>(),
                rt.params_ubo_ptrs[frame_idx],
                size_of::<RtParams>(),
            );
        }
        let params_gva = com::gpu_va(&rt.params_ubo_resources[frame_idx]);

        // SAFETY: the list is recording, and both buffers are live: the reset is a
        // 16-byte upload, the arguments rest in COPY_DEST.
        unsafe {
            cmd.CopyBufferRegion(&rt.ray_args, 0, &*rt.ray_args_reset, 0, RAY_ARGS_BYTES);
        }
        let mut before = vec![
            transition_barrier(
                &rt.ray_args,
                RAY_ARGS_REST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition_barrier(
                &targets.output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ];
        if let Some(p) = pyramid {
            before.push(transition_barrier(p.resource(), PYRAMID_REST, PYRAMID_READ));
        }
        // SAFETY: the list is recording, and every resource the barriers name is live.
        unsafe { cmd.ResourceBarrier(&before) };

        // SAFETY: the list is recording, and the root signature is owned by `rt`.
        unsafe { cmd.SetComputeRootSignature(&rt.root_sig) };
        // SAFETY: the list is recording, and the SRV heap is owned by `self.descriptors`.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: the signature declares a CBV at 0, and this frame's params buffer is live.
        unsafe { cmd.SetComputeRootConstantBufferView(0, params_gva) };
        // SAFETY: the SRV heap is set above, and the HDR target's descriptor is live.
        unsafe { cmd.set_compute_srv_table(5, self.targets.hdr.srv_gpu) };
        // SAFETY: the SRV heap is set above, and the gbuffer's normal-depth descriptor is live.
        unsafe { cmd.set_compute_srv_table(6, gbuffer.normal_depth_srv_gpu) };
        // SAFETY: the SRV heap is set above, and the gbuffer's roughness descriptor is live.
        unsafe { cmd.set_compute_srv_table(7, gbuffer.roughness_srv_gpu) };
        // SAFETY: the SRV heap is set above, and the prefiltered cube's descriptor is live.
        unsafe { cmd.set_compute_srv_table(8, self.prefilter_cube_srv_gpu()) };
        // Reflection-probe miss fallback: the cube array table at t10 + the
        // per-frame ProbeSet CBV at b4 + the per-frame records at t11, and
        // the main camera's cluster grid binning them (b5, t12). count == 0
        // keeps the sky path.
        self.probe_bindings(frame_idx).bind_compute(cmd, 12);
        // SAFETY: the SRV heap is set above, and the output's UAV descriptor is live.
        unsafe { cmd.set_compute_srv_table(ROOT_OUTPUT, targets.uav) };
        let pyramid_srv = pyramid.map_or(gbuffer.normal_depth_srv_gpu, PostTarget::srv_gpu);
        // SAFETY: the SRV heap is set above, and the pyramid's (or stand-in's) SRV is live.
        unsafe { cmd.set_compute_srv_table(ROOT_PYRAMID, pyramid_srv) };
        let list_gva = com::gpu_va(&targets.ray_list);
        let args_gva = com::gpu_va(&rt.ray_args);
        // SAFETY: the list is recording, and the ray list is owned by `rt` and in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(ROOT_RAY_LIST_UAV, list_gva) };
        // SAFETY: the list is recording, and the arguments were moved to UNORDERED_ACCESS above.
        unsafe { cmd.SetComputeRootUnorderedAccessView(ROOT_RAY_ARGS_UAV, args_gva) };
        // SAFETY: the list is recording, and the classify PSO is owned by `rt`.
        unsafe { cmd.SetPipelineState(&rt.classify_pso) };
        let (groups_x, groups_y) = classify_groups(rt.extent.0, rt.extent.1);
        // SAFETY: the pipeline and every root binding the classify reads are set above.
        unsafe { cmd.Dispatch(groups_x, groups_y, 1) };

        let mut listed = vec![
            uav_barrier(&targets.output),
            transition_barrier(
                &targets.ray_list,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                &rt.ray_args,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                RAY_ARGS_READ,
            ),
        ];
        if let Some(p) = pyramid {
            listed.push(transition_barrier(p.resource(), PYRAMID_READ, PYRAMID_REST));
        }
        // SAFETY: the list is recording, and every resource the barriers name is live.
        unsafe { cmd.ResourceBarrier(&listed) };

        // Textured hit shading needs the bindless albedo/normal pool, which only
        // the GPU-cull bindless path populates; otherwise fall back to the
        // flat-tint variant (mirrors Metal's bindless-arg-buffer gate).
        let textured = self.cull.main_bindless_pso.is_some();
        // Root SRVs: TLAS / vertex / index / geometry table (by GPU virtual
        // address; inline ray tracing reads the TLAS through a root SRV).
        // SAFETY: the list is recording, and the TLAS `accel` holds is live for the frame.
        unsafe { cmd.SetComputeRootShaderResourceView(1, accel.tlas_gva()) };
        let vertex_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
        // SAFETY: the list is recording, and the vertex buffer is owned by `self.scene.geometry`.
        unsafe { cmd.SetComputeRootShaderResourceView(2, vertex_gva) };
        let index_gva = com::gpu_va(&self.scene.geometry.index_buffer);
        // SAFETY: the list is recording, and the index buffer is owned by `self.scene.geometry`.
        unsafe { cmd.SetComputeRootShaderResourceView(3, index_gva) };
        // SAFETY: the list is recording, and the geometry table `accel` holds is live.
        unsafe { cmd.SetComputeRootShaderResourceView(4, accel.geom_table_gva()) };
        if textured {
            let pool = self.cull.bindless_pool_gpu[self.current_frame];
            // SAFETY: the textured PSO only runs when the bindless pool is populated, and this
            // frame's pool table is live.
            unsafe { cmd.set_compute_srv_table(9, pool) };
        }
        // Skinned-geometry root SRVs: the deformed (posed) vertex buffer +
        // the skinned index buffer the trace fetches a skinned hit from.
        // Both return a valid 1-element dummy GVA when there is no skinned
        // geometry, so the binding is always live.
        // SAFETY: the list is recording, and the deformed vertex GVA is always a live buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(10, accel.deformed_verts_gva()) };
        // SAFETY: the list is recording, and the skinned index GVA is always a live buffer.
        unsafe { cmd.SetComputeRootShaderResourceView(11, accel.skinned_index_gva()) };
        // The canopy block and view tables a hit on ground growing grass
        // shades from: this frame's copies, as the main pass binds them.
        // SAFETY: the list is recording, and this frame's canopy copy is live.
        unsafe { cmd.SetComputeRootConstantBufferView(17, self.canopy.gpu_va(frame_idx)) };
        // SAFETY: the list is recording, and this frame's canopy tables are live.
        unsafe { cmd.SetComputeRootShaderResourceView(18, self.canopy.tables_gpu_va(frame_idx)) };
        // SAFETY: the list is recording, and the ray list was moved to a shader-read state above.
        unsafe { cmd.SetComputeRootShaderResourceView(ROOT_RAY_LIST_SRV, list_gva) };
        // SAFETY: the list is recording, and the arguments were moved to a shader-read state above.
        unsafe { cmd.SetComputeRootShaderResourceView(ROOT_RAY_ARGS_SRV, args_gva) };
        // SAFETY: the list is recording, and both trace PSOs are owned by `rt`.
        unsafe {
            cmd.SetPipelineState(if textured {
                &rt.textured_pso
            } else {
                &rt.flat_pso
            });
        }
        // SAFETY: the pipeline and every root binding the trace reads are set above, and the
        // arguments hold the three group counts the classify grew to cover its list.
        unsafe { cmd.ExecuteIndirect(&rt.dispatch_signature, 1, &rt.ray_args, 0, None, 0) };

        let after = [
            transition_barrier(
                &targets.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                &targets.ray_list,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition_barrier(&rt.ray_args, RAY_ARGS_READ, RAY_ARGS_REST),
        ];
        // SAFETY: the list is recording, and every resource the barriers name is live.
        unsafe { cmd.ResourceBarrier(&after) };

        // Blur the reflection by roughness and composite it over the scene into the
        // reflection-composite output (the scene the post stack then consumes).
        self.encode_reflection_composite(cmd, frame_idx, targets.srv);
    }
}

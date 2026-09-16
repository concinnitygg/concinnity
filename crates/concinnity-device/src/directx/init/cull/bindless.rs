//! The GPU-driven main pass: its root signature, bucket 0's pipeline and the
//! material-referenced shader buckets, and the per-frame object buffers.

use concinnity_core::gfx::render_types;
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::{RenderError, RenderResult};
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use crate::directx::allocator::PooledBuffer;
use crate::directx::context::{FRAMES, align256, dump_on_err};
use crate::directx::error::map_hresult;
use crate::directx::init::InitGpu;
use crate::directx::init::pipelines::{
    BindlessMainShaders, BucketPipelineTargets, build_bucket_pipeline, build_world_pipeline_table,
    compile_main_bindless_shaders, create_main_bindless_root_signature,
};

pub(super) struct BindlessPass {
    pub(super) root_sig: ID3D12RootSignature,
    pub(super) pso: ID3D12PipelineState,
    pub(super) world_pipelines: Vec<Option<ID3D12PipelineState>>,
    pub(super) shaders: BindlessMainShaders,
    pub(super) object_buffers: Vec<PooledBuffer>,
    pub(super) object_ptrs: Vec<*mut u8>,
}

// The world's Shaders, one per bucket (`BackendInit::shaders`): entry 0 is the
// world default that drives bucket 0 (`programs: None` for the engine's own),
// entries 1.. are the material-referenced buckets. Each gets its own
// GPU-driven main-pass pipeline; an entry flagged `deferred` is a bucket whose
// Shader belongs to a scene that has not pinned, and is installed later by
// `install_world_shader`.
pub(super) fn build_bindless_pass(
    gpu: &InitGpu<'_>,
    world_shaders: &[WorldShader<'_>],
    plan: &CullPlan,
    msaa_samples: u32,
) -> RenderResult<BindlessPass> {
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let hot_reload = gpu.hot_reload;
    let n_cull = plan.n_cull;
    let world_default = world_shaders
        .first()
        .copied()
        .ok_or_else(|| RenderError::Other("BackendInit carried no shaders".to_string()))?;
    let bucket_shaders = world_shaders.get(1..).unwrap_or(&[]);
    // The GPU-driven main pass. The engine's pair is compiled regardless of the
    // world default: it is the program for every bucket that declares no Shader
    // and the source of the Wireframe twin. Bucket 0 takes the world default's
    // pair where the world declares one.
    let (bvs, bps) = compile_main_bindless_shaders(hot_reload)?;
    let bindless_main_shaders = BindlessMainShaders { vs: bvs, ps: bps };
    let brs = dump_on_err(info_queue, create_main_bindless_root_signature(device))?;
    let targets = BucketPipelineTargets {
        root_sig: &brs,
        msaa_samples,
        engine_default: &bindless_main_shaders,
        hot_reload,
    };
    let main_pso = build_bucket_pipeline(device, info_queue, targets, 0, world_default)?;

    // Material-referenced shaders (ShaderHandle 1..) each get their own
    // main-pass pipeline, so their draws route into their own region of the
    // GPU-culled command buffer.
    let world_pipelines = if bucket_shaders.is_empty() {
        Vec::new()
    } else {
        let max = render_types::MAX_SHADER_BUCKETS;
        if bucket_shaders.len() + 1 > max {
            return Err(RenderError::Other(format!(
                "world declares {} Shaders but at most {max} can be routed",
                bucket_shaders.len() + 1
            )));
        }
        build_world_pipeline_table(device, info_queue, targets, bucket_shaders)?
    };

    // Per-frame StructuredBuffer<GpuObjectData> upload buffers. Allocated only
    // when the world has anything to drive; rebuilt each frame in
    // `build_object_buffer`.
    let mut object_buffers: Vec<PooledBuffer> = Vec::new();
    let mut object_ptrs: Vec<*mut u8> = Vec::new();
    if n_cull > 0 {
        let object_buffer_size =
            align256((n_cull * std::mem::size_of::<render_types::GpuObjectData>()) as u64);
        // `FRAMES + 1`: the extra slot (index `FRAMES`) is reserved for the
        // asynchronous reflection-probe capture, which builds its CPU-written
        // bindless buffers into a slot the frame never touches (it uses
        // `[0, FRAMES)`). See `directx/probe.rs::bake_ring_slot`.
        for _ in 0..FRAMES + 1 {
            let buf = gpu.hw.alloc.alloc_buffer(
                object_buffer_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map object buffer"))?;
            object_ptrs.push(ptr as *mut u8);
            object_buffers.push(buf);
        }
    }

    Ok(BindlessPass {
        root_sig: brs,
        pso: main_pso,
        world_pipelines,
        shaders: bindless_main_shaders,
        object_buffers,
        object_ptrs,
    })
}

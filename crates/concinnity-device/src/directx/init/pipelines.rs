//! Main-pass pipeline construction shared by init and the runtime rebuilds:
//!   * Shader compilation for the bindless main pass and the GPU-driven shadow
//!     pass, from the program declarations in `directx/builtin_shaders.rs`.
//!   * Root-signature + PSO builders for the GPU-driven main pass, its shader
//!     buckets, and the depth-only shadow pass.
//!
//! Text + composite pipelines live in `directx/pipeline.rs`;
//! bloom/TAA/SSAO live in `directx/post/`; the GPU-cull compute pipeline lives
//! in `directx/cull.rs`.

use concinnity_core::render::backend_init;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shadow_bias;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::dump_on_err;
use crate::directx::pipeline::main_input_layout;
use crate::directx::pso::{Blend, Depth, DepthBias, GraphicsPso, Raster};
use crate::directx::root_sig::{Range, RootSig, Visibility};
use crate::directx::texture::HDR_FORMAT;

// Shader compilation

// The world Shader's program for `entry`, as a DXIL container: the cook's
// artifact when the engine template still matches, else a compile here. The
// DXIL pool is unbounded, so no pool size reaches the text.
pub(in crate::directx) fn world_entry(
    world: &concinnity_core::components::ShaderPrograms,
    entry: &str,
    hot_reload: bool,
) -> RenderResult<Vec<u8>> {
    let req = crate::shader::surface_source::Request {
        platform: concinnity_core::platform::Platform::DirectX,
        hot_reload,
    };
    crate::shader::surface_source::artifact(world, entry, &req, crate::shader::compile::cooked)
        .map(|c| c.into_owned())
}

// Compile the engine's bindless main-pass and pre-pass pairs. A bucket whose
// Shader is the world's compiles the same file through `world_entry` instead;
// the engine's pairs are the programs for every bucket that declares none, and
// the main pair the source of the Wireframe twin.
pub(in crate::directx) fn compile_main_bindless_shaders(
    hot_reload: bool,
) -> RenderResult<BindlessMainShaders> {
    Ok(BindlessMainShaders {
        vs: builtin_shaders::MAIN_BINDLESS_VERT.compile(hot_reload)?,
        ps: builtin_shaders::MAIN_BINDLESS_FRAG.compile(hot_reload)?,
        prepass_vs: builtin_shaders::MAIN_PREPASS_VERT.compile(hot_reload)?,
        prepass_ps: builtin_shaders::MAIN_PREPASS_FRAG.compile(hot_reload)?,
    })
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader. Built
// alongside the bindless main pass (same built-in-shader gate); a depth-only
// PSO with no pixel shader consumes it.
pub(in crate::directx) fn compile_shadow_bindless_vs(hot_reload: bool) -> RenderResult<Vec<u8>> {
    builtin_shaders::SHADOW_VERT_BINDLESS.compile(hot_reload)
}

// Root signature builders

// Root signature for the GPU-driven main pass, shared by every shader bucket.
//
// Slot [0] is a single-DWORD root constant carrying the per-draw object id
// (D3D12 `SV_InstanceID` does not include `StartInstanceLocation`, so the id
// rides a root constant); slot [5] is the unbounded bindless `Texture2D` pool
// (`t0, space1`); slot [8] is a root SRV at `t3` carrying the per-frame
// `StructuredBuffer<GpuObjectData>`.
pub(super) fn create_main_bindless_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use Visibility::{All, Pixel};
    RootSig::new()
        // [0] per-draw object id at b0 (1 DWORD).
        .constant_dwords(0, 1, All)
        // [1] view UBO at b1
        .cbv(1, All)
        // [2] light UBO at b2
        .cbv(2, Pixel)
        // [3] shadow UBO at b3
        .cbv(3, Pixel)
        // [4] shadow map array (t0) + IBL cubes (t5..t6)
        .table(&[Range::srv(0, 1), Range::srv(5, 2)], Pixel)
        // [5] Unbounded bindless pool: `Texture2D tex_pool[] : register(t0,
        // space1)`. The table base GPU handle points at the per-object SRV region
        // (heap slot `object_base_slot`), so pool index `2*i` / `2*i+1` resolves
        // to object `i`'s albedo / normal SRV.
        .table(&[Range::bindless_srv(1)], Pixel)
        // [6] shadow comparison sampler (s0)
        .sampler_table(0, 1, Pixel)
        // [7] linear repeat (s1) + cube sampler (s2)
        .sampler_table(1, 2, Pixel)
        // [8] per-frame StructuredBuffer<GpuObjectData> at t3
        .srv(3, All)
        // [9] SSAO occlusion SRV (t4)
        .srv_table(4, 1, Pixel)
        // [10] the reflection-probe cube array at t7 (`TextureCubeArray
        // probe_cubes : register(t7)`), one descriptor however many cubes it holds.
        .srv_table(7, 1, Pixel)
        // [11] the ProbeSet (live probe count) at b4.
        .cbv(4, Pixel)
        // [12] per-scene StructuredBuffer<GpuLight> at t1 (matches
        // main_bindless.hlsl's CN_BACKEND_DIRECTX block).
        .srv(1, Pixel)
        // [13] ClusterParams at b5 (b4 is the ProbeSet cbuffer).
        .cbv(5, Pixel)
        // [14] per-cluster light-index lists at t2.
        .srv(2, Pixel)
        // [15] per-slice StructuredBuffer<SpotShadowData> at t15.
        .srv(15, Pixel)
        // [16] spot shadow depth array at t16, one register past the spot shadow
        // records at t15.
        .srv_table(16, 1, Pixel)
        // [17] per-scene StructuredBuffer<AreaLightData> at t17.
        .srv(17, Pixel)
        // [18] the area-light LTC tables at t18..t19, past the spot shadow array.
        .srv_table(18, 2, Pixel)
        // [19] the reflection-probe records at t8.
        .srv(8, Pixel)
        // [20] the material parameter table at t20, which either world hook may
        // read.
        .srv(20, All)
        .input_layout()
        .build(device, "main bindless root sig")
}

// Root signature for the GPU-driven shadow pass's depth-only bindless pipeline.
// Mirrors the bindless main root signature's object-id delivery so the shared
// cull command signature works against it: [0] is the per-command b0 object-id
// root constant (set by the `ExecuteIndirect` command signature, so it MUST stay
// at root parameter 0), [1] the shadow UBO CBV (light_vps), [2] a per-cascade b2
// cascade-index root constant (set once per cascade's `ExecuteIndirect`), and [3]
// the per-frame `StructuredBuffer<GpuObjectData>` root SRV the VS reads `model`
// from. All vertex-stage only (depth-only pass, no pixel shader).
pub(in crate::directx) fn create_shadow_bindless_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        // [0] b0: object id (set per command by the command sig).
        .constant_dwords(0, 1, Visibility::Vertex)
        // [1] b1: shadow UBO (light_vps[4] + cascade_splits).
        .cbv(1, Visibility::Vertex)
        // [2] b2: cascade index (set per cascade's ExecuteIndirect).
        .constant_dwords(2, 1, Visibility::Vertex)
        // [3] t0: per-frame StructuredBuffer<GpuObjectData>.
        .srv(0, Visibility::Vertex)
        .input_layout()
        .build(device, "shadow bindless root sig")
}

// PSO builders

// PSO for the GPU-driven main pass: one per shader bucket, all against the
// bindless root signature.
pub(in crate::directx) fn create_main_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> RenderResult<ID3D12PipelineState> {
    create_main_pso_filled(device, root_sig, vs, ps, rtv_format, sample_count, false)
}

// The Wireframe view mode's variant of `create_main_pso`. D3D12 fill mode is
// pipeline state (unlike Metal's encoder flag), so the mode needs its own PSO;
// see [`super::super::wireframe`].
pub(in crate::directx) fn create_main_pso_wireframe(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> RenderResult<ID3D12PipelineState> {
    create_main_pso_filled(device, root_sig, vs, ps, rtv_format, sample_count, true)
}

fn create_main_pso_filled(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
    wireframe: bool,
) -> RenderResult<ID3D12PipelineState> {
    let layout = main_input_layout();
    GraphicsPso::new(root_sig, vs, ps)
        .input_layout(&layout)
        .target(rtv_format, Blend::Opaque)
        .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
        .samples(sample_count)
        // No culling, matching Metal's default, so meshes with mixed winding
        // (e.g. procedural floor/ceiling planes) render from both sides.
        .raster(Raster {
            wireframe,
            ..Raster::default()
        })
        .build(device, "main")
}

pub(in crate::directx) fn create_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    let layout = main_input_layout();
    GraphicsPso::new(root_sig, vs, &[])
        .input_layout(&layout)
        .depth(DXGI_FORMAT_D32_FLOAT, Depth::write())
        // No culling, matching Metal, so double-sided procedural meshes cast
        // shadows correctly.
        .raster(Raster {
            bias: DepthBias {
                constant: shadow_bias::RASTER_CONSTANT as i32,
                clamp: shadow_bias::RASTER_CLAMP,
                slope: shadow_bias::RASTER_SLOPE,
            },
            ..Raster::default()
        })
        .build(device, "shadow")
}

// Material-referenced world shader pipelines

// The engine's own compiled bindless main-pass and pre-pass stages, kept past
// init so a shader bucket that resolves to the engine default can build its
// pipelines without recompiling, and so the Wireframe twin has its source.
// Recompiling cost ~140 ms per bucket install, which is the whole point of
// warming a pipeline behind a loading screen.
pub(in crate::directx) struct BindlessMainShaders {
    pub vs: Vec<u8>,
    pub ps: Vec<u8>,
    pub prepass_vs: Vec<u8>,
    pub prepass_ps: Vec<u8>,
}

// One shader bucket's PSOs: the main pass that shades its draws, and the
// G-buffer pre-pass that lays down their depth, normal, roughness and motion
// from the same vertex hook.
pub(in crate::directx) struct BucketPsos {
    pub main: ID3D12PipelineState,
    // `None` until the world has a G-buffer, or after this bucket's pre-pass
    // failed to build; the pre-pass then skips the bucket's draws.
    pub prepass: Option<ID3D12PipelineState>,
    // The world Shader the bucket compiles, `None` for the engine's own
    // programs, kept so the pre-pass can be built or rebuilt later.
    pub programs: Option<concinnity_core::components::ShaderPrograms>,
}

// The root signatures a bucket's PSOs bind: the main pass's, and the G-buffer
// pre-pass's once the world has a G-buffer, which is what decides whether a
// bucket gets a pre-pass PSO at all.
#[derive(Clone, PartialEq)]
pub(in crate::directx) struct BucketRootSigs {
    pub main: ID3D12RootSignature,
    pub prepass: Option<ID3D12RootSignature>,
}

// Build one shader bucket's PSOs. `bucket` is the `DrawObject::shader_bucket`
// value (bucket 0 is the world default program) and names the bucket in error
// messages.
//
// A bucket with no programs is one the world declared no Shader for, so the
// engine's own bindless programs render it.
pub(in crate::directx) fn build_bucket_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: BucketPipelineTargets<'_>,
    bucket: usize,
    shader: backend_init::WorldShader<'_>,
) -> RenderResult<BucketPsos> {
    match shader.programs {
        Some(programs) => {
            build_world_shader_pso(device, info_queue, targets.world(), bucket, programs)
        }
        None => {
            let engine = targets.engine_default;
            create_bucket_psos(
                device,
                info_queue,
                targets.world(),
                bucket,
                BucketStages {
                    vs: &engine.vs,
                    ps: &engine.ps,
                    prepass_vs: &engine.prepass_vs,
                    prepass_ps: &engine.prepass_ps,
                },
            )
        }
    }
}

// Build a world Shader's PSOs for bucket `bucket` from its own compiled stages.
pub(in crate::directx) fn build_world_shader_pso(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: WorldPsoTargets<'_>,
    bucket: usize,
    programs: &concinnity_core::components::ShaderPrograms,
) -> RenderResult<BucketPsos> {
    use concinnity_core::render::shader_programs::surface;
    let entry =
        |program: surface::Program| world_entry(programs, program.entry, targets.hot_reload);
    let vs = entry(surface::MAIN_VERTEX)?;
    let ps = entry(surface::MAIN_FRAGMENT)?;
    // A pre-pass that does not compile costs the bucket its G-buffer draws,
    // not its shading: `create_bucket_psos` warns over the empty stages.
    let (prepass_vs, prepass_ps) = match targets.root_sigs.prepass {
        Some(_) => world_prepass_stages(programs, targets.hot_reload, bucket),
        None => (Vec::new(), Vec::new()),
    };
    let mut psos = create_bucket_psos(
        device,
        info_queue,
        targets,
        bucket,
        BucketStages {
            vs: &vs,
            ps: &ps,
            prepass_vs: &prepass_vs,
            prepass_ps: &prepass_ps,
        },
    )?;
    psos.programs = Some(programs.clone());
    Ok(psos)
}

// A world Shader's pre-pass stages, or empty ones with a warning when they do
// not compile.
fn world_prepass_stages(
    programs: &concinnity_core::components::ShaderPrograms,
    hot_reload: bool,
    bucket: usize,
) -> (Vec<u8>, Vec<u8>) {
    use concinnity_core::render::shader_programs::surface;
    let stages = world_entry(programs, surface::PREPASS_VERTEX.entry, hot_reload).and_then(|vs| {
        world_entry(programs, surface::PREPASS_FRAGMENT.entry, hot_reload).map(|ps| (vs, ps))
    });
    stages.unwrap_or_else(|e| {
        tracing::warn!("shader bucket {bucket}'s G-buffer pre-pass did not build: {e}");
        (Vec::new(), Vec::new())
    })
}

// Bucket `bucket`'s pre-pass PSO over `root_sig` from its compiled stages, or
// `None` with a warning when it cannot be built: the bucket then shades
// without contributing to the G-buffer.
pub(in crate::directx) fn build_bucket_prepass(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    root_sig: &ID3D12RootSignature,
    bucket: usize,
    stages: (&[u8], &[u8]),
) -> Option<ID3D12PipelineState> {
    let (vs, ps) = stages;
    if vs.is_empty() || ps.is_empty() {
        return None;
    }
    let pso = dump_on_err(
        info_queue,
        crate::directx::post::gbuffer::create_prepass_pso(device, root_sig, vs, ps),
    );
    pso.map_err(|e| tracing::warn!("shader bucket {bucket}'s G-buffer pre-pass did not build: {e}"))
        .ok()
}

// Bucket `bucket`'s pre-pass PSO from the engine's programs or `world`'s,
// for a bucket whose main PSO already exists.
pub(in crate::directx) fn build_prepass_for(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    root_sig: &ID3D12RootSignature,
    bucket: usize,
    source: PrepassSource<'_>,
) -> Option<ID3D12PipelineState> {
    let (vs, ps) = match source {
        PrepassSource::World(programs, hot_reload) => {
            world_prepass_stages(programs, hot_reload, bucket)
        }
        PrepassSource::Engine(engine) => (engine.prepass_vs.clone(), engine.prepass_ps.clone()),
    };
    build_bucket_prepass(device, info_queue, root_sig, bucket, (&vs, &ps))
}

// Where a bucket's pre-pass stages come from.
pub(in crate::directx) enum PrepassSource<'a> {
    World(&'a concinnity_core::components::ShaderPrograms, bool),
    Engine(&'a BindlessMainShaders),
}

// The four compiled stages one bucket's PSOs take.
struct BucketStages<'a> {
    vs: &'a [u8],
    ps: &'a [u8],
    prepass_vs: &'a [u8],
    prepass_ps: &'a [u8],
}

fn create_bucket_psos(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: WorldPsoTargets<'_>,
    bucket: usize,
    stages: BucketStages<'_>,
) -> RenderResult<BucketPsos> {
    let BucketStages {
        vs,
        ps,
        prepass_vs,
        prepass_ps,
    } = stages;
    if vs.is_empty() || ps.is_empty() {
        return Err(RenderError::ShaderCompile(format!(
            "shader bucket {bucket} carries no vertex/fragment bytecode"
        )));
    }
    let main = dump_on_err(
        info_queue,
        create_main_pso(
            device,
            &targets.root_sigs.main,
            vs,
            ps,
            HDR_FORMAT,
            targets.msaa_samples,
        ),
    )
    .map_err(|e| e.context(format!("shader bucket {bucket}")))?;
    let prepass = targets.root_sigs.prepass.as_ref().and_then(|root_sig| {
        build_bucket_prepass(
            device,
            info_queue,
            root_sig,
            bucket,
            (prepass_vs, prepass_ps),
        )
    });
    Ok(BucketPsos {
        main,
        prepass,
        programs: None,
    })
}

// What every bucket pipeline shares: the root signatures it binds against, the
// sample count, and the engine's own programs for a bucket with none.
#[derive(Clone, Copy)]
pub(in crate::directx) struct BucketPipelineTargets<'a> {
    pub root_sigs: &'a BucketRootSigs,
    pub msaa_samples: u32,
    pub engine_default: &'a BindlessMainShaders,
    pub hot_reload: bool,
}

impl<'a> BucketPipelineTargets<'a> {
    fn world(self) -> WorldPsoTargets<'a> {
        WorldPsoTargets {
            root_sigs: self.root_sigs,
            msaa_samples: self.msaa_samples,
            hot_reload: self.hot_reload,
        }
    }
}

// What a world Shader's own pipelines are built against.
#[derive(Clone, Copy)]
pub(in crate::directx) struct WorldPsoTargets<'a> {
    pub root_sigs: &'a BucketRootSigs,
    pub msaa_samples: u32,
    pub hot_reload: bool,
}

// Build the per-bucket pipeline table from the world's material-referenced
// shaders. Index `b` holds bucket `b + 1`'s pipeline; `None` marks a bucket the
// streaming pump installs later (its Shader is owned by a scene that has not
// pinned, so `decode_shaders` handed over an all-empty payload).
pub(super) fn build_world_pipeline_table(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: BucketPipelineTargets<'_>,
    bucket_shaders: &[backend_init::WorldShader<'_>],
) -> RenderResult<Vec<Option<BucketPsos>>> {
    let mut table = Vec::with_capacity(bucket_shaders.len());
    for (i, shader) in bucket_shaders.iter().enumerate() {
        let bucket = i + 1;
        // A bucket whose Shader a non-start scene owns has no payload yet; the
        // streaming pump installs it when that scene pins.
        if shader.deferred {
            table.push(None);
            continue;
        }
        table.push(Some(build_bucket_pipeline(
            device, info_queue, targets, bucket, *shader,
        )?));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    // The bindless main pair compiles from `main_bindless.hlsl` (dxc, DXIL sm
    // 6.0). This compiles it offline so a syntax or register error fails a test
    // instead of only surfacing as an init failure on a GPU host.
    #[test]
    fn bindless_main_shaders_compile() {
        concinnity_shader::require_dxc!();
        super::compile_main_bindless_shaders(false).expect("bindless main shaders must compile");
    }
}

//! Main-pass pipeline construction shared by init and the runtime rebuilds:
//!   * Shader compilation for the bindless main pass and the GPU-driven shadow
//!     pass, from the program declarations in `directx/slang_builtins.rs`.
//!   * Root-signature + PSO builders for the GPU-driven main pass, its shader
//!     buckets, and the depth-only shadow pass.
//!
//! Text + composite pipelines live in `directx/pipeline.rs`;
//! bloom/TAA/SSAO live in `directx/post/`; the GPU-cull compute pipeline lives
//! in `directx/cull.rs`; the skinned shadow pipeline (built lazily once a
//! `SkinnedMesh` is uploaded) lives in `directx/resources.rs`.

use concinnity_core::render::backend_init;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shadow_bias;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::com;
use crate::directx::context::dump_on_err;
use crate::directx::draw::shadow::ShadowPush;
use crate::directx::error::map_pso_hresult;
use crate::directx::pipeline::{main_input_layout, serialize_and_create_root_sig};
use crate::directx::root_constants::root_dwords;
use crate::directx::slang_builtins;
use crate::directx::slang_builtins::SlangCompile;
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
        platform: concinnity_core::platform::Platform::Hlsl,
        probe_count: concinnity_core::render::uniforms::MAX_PROBES,
        hot_reload,
    };
    crate::shader::surface_source::artifact(world, entry, &req)
        .map(|c| c.into_owned())
        .map_err(RenderError::ShaderCompile)
}

// Compile the engine's bindless static-pass pair. A bucket whose Shader is the
// world's compiles the same file through `world_entry` instead; the engine's
// pair is the program for every bucket that declares none and the source of
// the Wireframe twin.
pub(in crate::directx) fn compile_main_bindless_shaders(
    hot_reload: bool,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = slang_builtins::MAIN_BINDLESS_VERT.compile(hot_reload)?;
    let ps = slang_builtins::MAIN_BINDLESS_FRAG.compile(hot_reload)?;
    Ok((vs, ps))
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader. Built
// alongside the bindless main pass (same built-in-shader gate); a depth-only
// PSO with no pixel shader consumes it.
pub(in crate::directx) fn compile_shadow_bindless_vs(hot_reload: bool) -> RenderResult<Vec<u8>> {
    slang_builtins::SHADOW_BINDLESS_VERT.compile(hot_reload)
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
    let shadow_srv_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0, // t0
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 5, // t5..t6
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    // Unbounded bindless pool: `Texture2D tex_pool[] : register(t0, space1)`.
    // The table base GPU handle points at the per-object SRV region (heap slot
    // `object_base_slot`), so pool index `2*i` / `2*i+1` resolves to object
    // `i`'s albedo / normal SRV.
    let pool_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: u32::MAX, // unbounded
        BaseShaderRegister: 0,    // t0
        RegisterSpace: 1,         // space1
        OffsetInDescriptorsFromTableStart: 0,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // s0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let linear_cube_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // s1..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [9] table: SSAO occlusion SRV at t4.
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [10] table: the reflection-probe cube array at t7..t7+MAX_PROBES
    // (`TextureCube probe_cubes[MAX_PROBES] : register(t7)`). Unbaked slots hold the sky
    // prefilter cube, so a sample at any index is always valid.
    let probe_cube_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: concinnity_core::render::uniforms::MAX_PROBES as u32,
        BaseShaderRegister: 7, // t7..
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [16] table: spot shadow depth array at t16, one register past the spot
    // shadow records at t15 that follow the probe cube array (t7..t7+MAX_PROBES).
    let spot_shadow_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 7 + concinnity_core::render::uniforms::MAX_PROBES as u32 + 1, // t16
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [18] table: the area-light LTC tables, past the spot shadow array.
    let ltc_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 7 + concinnity_core::render::uniforms::MAX_PROBES as u32 + 3, // t18..t19
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constant: per-draw object id at b0 (1 DWORD).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root CBV: view UBO at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root CBV: light UBO at b2
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Root CBV: shadow UBO at b3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [4] Descriptor table: shadow map array (t0) + IBL cubes (t5..t6)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: shadow_srv_ranges.len() as u32,
                    pDescriptorRanges: shadow_srv_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [5] Descriptor table: unbounded bindless texture pool (t0, space1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &pool_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [6] Descriptor table: shadow comparison sampler (s0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &shadow_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [7] Descriptor table: linear repeat (s1) + cube sampler (s2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &linear_cube_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [8] Root SRV: per-frame StructuredBuffer<GpuObjectData> at t3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [9] Descriptor table: SSAO occlusion SRV (t4)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ssao_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [10] Descriptor table: reflection-probe cube array (t7..)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &probe_cube_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [11] Root CBV: the ProbeSet (parallax boxes + live count) at b4.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 4, // b4
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [12] Root SRV: per-scene StructuredBuffer<GpuLight> at t1 (matches
        // main_bindless.slang's DXIL_ABI block).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [13] Root CBV: ClusterParams at b5 (b4 is the ProbeSet cbuffer).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 5,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [14] Root SRV: per-cluster light-index lists at t2 (t7.. is the probe
        // cube array).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [15] Root SRV: per-slice StructuredBuffer<SpotShadowData>, past the
        // probe cube array at t15.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 7 + concinnity_core::render::uniforms::MAX_PROBES as u32, // t15
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [16] table: spot shadow depth array at t16.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &spot_shadow_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [17] Root SRV: per-scene StructuredBuffer<AreaLightData> at t17.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 7 + concinnity_core::render::uniforms::MAX_PROBES as u32 + 2, // t17,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [18] table: the area-light LTC tables at t18..t19.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ltc_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    serialize_and_create_root_sig(device, &params, "main bindless root sig")
}

pub(in crate::directx) fn create_shadow_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    let params = [
        // [0] Root constants: `ShadowPush` at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: root_dwords::<ShadowPush>(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV: shadow UBO (light_vps[4] + cascade_splits) at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
    ];

    serialize_and_create_root_sig(device, &params, "shadow root sig")
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
    let params = [
        // [0] Root constant b0: object id (set per command by the command sig).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV b1: shadow UBO (light_vps[4] + cascade_splits).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [2] Root constant b2: cascade index (set per cascade's ExecuteIndirect).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [3] Root SRV t0: per-frame StructuredBuffer<GpuObjectData>.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
    ];

    serialize_and_create_root_sig(device, &params, "shadow bindless root sig")
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
    create_main_pso_filled(
        device,
        root_sig,
        vs,
        ps,
        rtv_format,
        sample_count,
        D3D12_FILL_MODE_SOLID,
    )
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
    create_main_pso_filled(
        device,
        root_sig,
        vs,
        ps,
        rtv_format,
        sample_count,
        D3D12_FILL_MODE_WIREFRAME,
    )
}

fn create_main_pso_filled(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
    fill_mode: D3D12_FILL_MODE,
) -> RenderResult<ID3D12PipelineState> {
    let layout = main_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = rtv_format;
            a
        },
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: fill_mode,
            // Match Metal's default (no culling) so meshes with mixed winding
            // (e.g. procedural floor/ceiling planes) render from both sides.
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: {
                let mut arr = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
                arr[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };

    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    unsafe { crate::directx::pso_library::create_graphics(device, &pso_desc) }
        .map_err(|e| map_pso_hresult(e.code(), "create main PSO"))
}

pub(in crate::directx) fn create_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    let layout = main_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            // Match Metal: shadow pass also uses no culling so double-sided
            // procedural meshes cast shadows correctly.
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: shadow_bias::RASTER_CONSTANT as i32,
            DepthBiasClamp: shadow_bias::RASTER_CLAMP,
            SlopeScaledDepthBias: shadow_bias::RASTER_SLOPE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            ..Default::default()
        },
        ..Default::default()
    };

    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    unsafe { crate::directx::pso_library::create_graphics(device, &pso_desc) }
        .map_err(|e| map_pso_hresult(e.code(), "create shadow PSO"))
}

// Material-referenced world shader pipelines

// The engine's own compiled bindless main-pass stages, kept past init so a
// shader bucket that resolves to the engine default can build its pipeline
// without recompiling, and so the Wireframe twin has its source. Recompiling
// cost ~140 ms per bucket install, which is the whole point of warming a
// pipeline behind a loading screen.
pub(in crate::directx) struct BindlessMainShaders {
    pub vs: Vec<u8>,
    pub ps: Vec<u8>,
}

// Build one shader bucket's bindless main-pass pipeline. `bucket` is the
// `DrawObject::shader_bucket` value (1-based; bucket 0 is the world default
// program) and names the bucket in error messages.
//
// A bucket with no programs is one the world declared no Shader for, so the
// engine's own bindless program renders it.
pub(in crate::directx) fn build_bucket_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: BucketPipelineTargets<'_>,
    bucket: usize,
    shader: backend_init::WorldShader<'_>,
) -> RenderResult<ID3D12PipelineState> {
    let (vs, ps) = match shader.programs {
        Some(programs) => (
            world_entry(programs, "vertex_main_bindless", targets.hot_reload)?,
            world_entry(programs, "fragment_main_bindless", targets.hot_reload)?,
        ),
        None => (
            targets.engine_default.vs.clone(),
            targets.engine_default.ps.clone(),
        ),
    };
    if vs.is_empty() || ps.is_empty() {
        return Err(RenderError::ShaderCompile(format!(
            "shader bucket {bucket} carries no vertex/fragment bytecode"
        )));
    }
    dump_on_err(
        info_queue,
        create_main_pso(
            device,
            targets.root_sig,
            &vs,
            &ps,
            HDR_FORMAT,
            targets.msaa_samples,
        ),
    )
    .map_err(|e| e.context(format!("shader bucket {bucket}")))
}

// What every bucket pipeline shares: the bindless root signature it binds
// against, the sample count, and the engine's own pair for a bucket with no
// programs.
#[derive(Clone, Copy)]
pub(in crate::directx) struct BucketPipelineTargets<'a> {
    pub root_sig: &'a ID3D12RootSignature,
    pub msaa_samples: u32,
    pub engine_default: &'a BindlessMainShaders,
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
) -> RenderResult<Vec<Option<ID3D12PipelineState>>> {
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
    // The bindless main pair compiles from `src/shaders/main_bindless.slang` at
    // runtime (slangc, DXIL sm 6.0). This compiles it offline so a syntax or
    // register error fails a test instead of only surfacing as an init failure
    // on a GPU host.
    #[test]
    fn bindless_main_shaders_compile() {
        if !concinnity_slang::shader_tests_enabled() {
            return;
        }
        super::compile_main_bindless_shaders(false).expect("bindless main shaders must compile");
    }
}

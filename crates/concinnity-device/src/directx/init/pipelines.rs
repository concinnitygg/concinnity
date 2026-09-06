// src/directx/init/pipelines.rs
//
// Core render-pipeline construction extracted from DxContext::new:
//   * Shader compilation (`compile_all_shaders`, `compile_main_bindless_shaders`),
//     from the program declarations in `directx/slang_builtins.rs`.
//   * Root-signature + PSO builders for the GPU-driven main pass, its shader
//     buckets, and the depth-only shadow pass.
//   * High-level `build_main_pipelines`/`build_shadow_pipeline`/etc.
//     orchestration helpers consumed by init/mod.rs.
//
// Mirrors src/metal/init/pipelines.rs (the same set of pipelines built at
// init time). Text + composite pipelines live in `directx/pipeline.rs`;
// bloom/TAA/SSAO live in `directx/post/`; the GPU-cull compute pipeline lives
// in `directx/cull.rs`; the skinned shadow pipeline (built lazily once a
// `SkinnedMesh` is uploaded) lives in `directx/resources.rs`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::com;
use crate::directx::context::{FRAMES, align256, dump_on_err};
use crate::directx::cull::{
    INDIRECT_COMMAND_STRIDE, compile_cull_shader, compile_cull_shader_phase2,
    compile_cull_shader_shadow, create_cull_command_signature, create_cull_pso,
    create_cull_root_signature,
};
use crate::directx::pipeline::{
    compile_composite_shaders, compile_text_shaders, create_composite_pso,
    create_composite_root_signature, create_text_pso, create_text_root_signature,
    main_input_layout, serialize_and_create_root_sig,
};
use crate::directx::slang_builtins;
use crate::directx::slang_builtins::SlangCompile;
use crate::directx::texture::{HDR_FORMAT, create_buffer, create_uav_buffer};
use crate::gfx::shadow_bias;

// Shader compilation

pub(super) struct CompiledShaders {
    pub shadow_vs: Option<Vec<u8>>,
    pub text_vs: Vec<u8>,
    pub text_ps: Vec<u8>,
}

// The world Shader's program for `entry`, as a DXIL container: the cook's
// artifact when the engine template still matches, else a compile here. The
// DXIL pool is unbounded, so no pool size reaches the text.
pub(in crate::directx) fn world_entry(
    world: &concinnity_core::components::ShaderPrograms,
    entry: &str,
    hot_reload: bool,
) -> Result<Vec<u8>, String> {
    let req = crate::surface_source::Request {
        platform: concinnity_core::platform::Platform::Hlsl,
        pool_size: 0,
        probe_count: concinnity_core::render::uniforms::MAX_PROBES,
        hot_reload,
    };
    crate::surface_source::artifact(world, entry, &req).map(|c| c.into_owned())
}

// Compile the engine-internal stages the init path needs outside the
// GPU-driven main pass.
pub(super) fn compile_all_shaders(hot_reload: bool) -> Result<CompiledShaders, String> {
    // Whether the shadow pass runs is gated by `effective_shadow_size` at the
    // call site.
    let shadow_vs = Some(slang_builtins::SHADOW_VERT.compile(hot_reload)?);
    let (text_vs, text_ps) = compile_text_shaders(hot_reload)?;
    Ok(CompiledShaders {
        shadow_vs,
        text_vs,
        text_ps,
    })
}

// Compile the engine's bindless static-pass pair. A bucket whose Shader is the
// world's compiles the same file through `world_entry` instead; the engine's
// pair is the program for every bucket that declares none and the source of
// the Wireframe twin.
pub(in crate::directx) fn compile_main_bindless_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vs = slang_builtins::MAIN_BINDLESS_VERT.compile(hot_reload)?;
    let ps = slang_builtins::MAIN_BINDLESS_FRAG.compile(hot_reload)?;
    Ok((vs, ps))
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader. Built
// alongside the bindless main pass (same built-in-shader gate); a depth-only
// PSO with no pixel shader consumes it.
pub(in crate::directx) fn compile_shadow_bindless_vs(hot_reload: bool) -> Result<Vec<u8>, String> {
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
fn create_main_bindless_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
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
    // [16] table: spot shadow depth array. The probe cube array above runs
    // t7..t7+MAX_PROBES, so this clears it at t15/t16 rather than reusing the
    // legacy shader's t10.
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
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root constants: model mat4 (16) + cascade_idx + 3 pad = 20 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 20,
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
) -> Result<ID3D12RootSignature, String> {
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
) -> Result<ID3D12PipelineState, String> {
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
) -> Result<ID3D12PipelineState, String> {
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
) -> Result<ID3D12PipelineState, String> {
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
        .map_err(|e| format!("create main PSO: {e}"))
}

pub(in crate::directx) fn create_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> Result<ID3D12PipelineState, String> {
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
        .map_err(|e| format!("create shadow PSO: {e}"))
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
    shader: crate::gfx::backend_init::WorldShader<'_>,
) -> Result<ID3D12PipelineState, String> {
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
        return Err(format!(
            "shader bucket {bucket} carries no vertex/fragment bytecode"
        ));
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
    .map_err(|e| format!("shader bucket {bucket}: {e}"))
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
fn build_world_pipeline_table(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    targets: BucketPipelineTargets<'_>,
    bucket_shaders: &[crate::gfx::backend_init::WorldShader<'_>],
) -> Result<Vec<Option<ID3D12PipelineState>>, String> {
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

// Init-time orchestration

pub(super) struct MainPipelines {
    // The GPU-driven main pass's root signature and bucket 0's PSO: the world
    // default Shader's pair where the world declares one, the engine's pair
    // otherwise.
    pub main_bindless_root_sig: Option<ID3D12RootSignature>,
    pub main_bindless_pso: Option<ID3D12PipelineState>,
    // Material-referenced world shader pipelines, indexed by `shader_bucket - 1`.
    // Empty unless the world declares more than one Shader.
    pub world_pipelines: Vec<Option<ID3D12PipelineState>>,
    // Commands reserved per bucket region in the indirect buffers.
    pub bucket_stride: usize,
    // The engine's compiled bindless main-pass stages, retained for the buckets a
    // scene warms mid-session and for the Wireframe twin.
    pub bindless_main_shaders: BindlessMainShaders,
    pub object_buffer_resources: Vec<PooledBuffer>,
    pub object_buffer_ptrs: Vec<*mut u8>,
    pub cull_root_sig: Option<ID3D12RootSignature>,
    pub cull_pso: Option<ID3D12PipelineState>,
    // Phase-2 cull PSO for two-pass occlusion (`main_phase2` entry, same root
    // signature as `cull_pso`). `Some` only when the world requested
    // `occlusion_two_pass` AND the bindless cull path is active.
    pub cull_pso_phase2: Option<ID3D12PipelineState>,
    pub cull_command_signature: Option<ID3D12CommandSignature>,
    pub draw_args_buffer_resources: Vec<PooledBuffer>,
    pub draw_args_buffer_ptrs: Vec<*mut u8>,
    pub indirect_cmd_buffers: Vec<ID3D12Resource>,
    // Per-frame per-object cull-status buffers (one u32 each). Phase-1 cull
    // writes drawn / hi-z-candidate / culled; phase-2 cull reads it. Always
    // allocated when the bindless cull path is active (mirrors Metal, where the
    // status buffer is always present and ignored under single-pass).
    pub cull_status_buffers: Vec<ID3D12Resource>,
    // Per-frame second indirect-command buffers the phase-2 cull writes and
    // `Main2` consumes. `Some`/non-empty only under two-pass occlusion.
    pub indirect_cmd_buffers_2: Vec<ID3D12Resource>,
    // GPU-driven shadow pass. Depth-only bindless pipeline + the
    // shared cull command signature rebuilt against its root sig + per-frame
    // indirect buffers (one region per cascade) + a scratch cull-status buffer.
    // All `Some`/non-empty only when the bindless cull path is active AND shadows
    // are enabled.
    pub shadow_bindless_root_sig: Option<ID3D12RootSignature>,
    pub shadow_bindless_pso: Option<ID3D12PipelineState>,
    pub shadow_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    // Frustum-only shadow cull PSO (`main_shadow` entry, shares the cull root sig).
    pub cull_pso_shadow: Option<ID3D12PipelineState>,
    pub shadow_indirect_buffers: Vec<ID3D12Resource>,
    pub shadow_cull_status_buffers: Vec<ID3D12Resource>,
    // GPU-driven G-buffer pre-pass. A 3-MRT bindless pipeline + the
    // shared cull command signature rebuilt against its root sig + per-frame
    // previous-frame model upload buffers. All `Some`/non-empty only when the
    // bindless cull path is active AND the G-buffer is enabled.
    pub gbuffer_bindless_root_sig: Option<ID3D12RootSignature>,
    pub gbuffer_bindless_pso: Option<ID3D12PipelineState>,
    pub gbuffer_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    pub prev_model_buffer_resources: Vec<PooledBuffer>,
    pub prev_model_buffer_ptrs: Vec<*mut u8>,
}

// The world's Shaders, one per bucket (`BackendInit::shaders`): entry 0 is the
// world default that drives bucket 0 (`programs: None` for the engine's own),
// entries 1.. are the material-referenced buckets. Each gets its own
// GPU-driven main-pass pipeline; an entry flagged `deferred` is a bucket whose
// Shader belongs to a scene that has not pinned, and is installed later by
// `install_world_shader`.
#[derive(Clone, Copy)]
pub(super) struct MainPipelineShaders<'a> {
    pub world_shaders: &'a [crate::gfx::backend_init::WorldShader<'a>],
}

// Record counts + MSAA that size the GPU-driven bindless pass's cull / object /
// draw-args / indirect buffers.
#[derive(Clone, Copy)]
pub(super) struct MainPipelineConfig {
    // Static build-time object count.
    pub n_objects: usize,
    // Total instanced-cluster instances folded in after the static objects.
    pub n_instances: usize,
    // Skinned draw objects folded in after the instances.
    pub n_skinned: usize,
    // Worst-case resident chunk count for a streaming VoxelWorld (0 otherwise),
    // reserved between the instances and the skinned tail.
    pub n_chunk_max: usize,
    // MSAA sample count for the HDR render target.
    pub msaa_samples: u32,
}

// Which optional GPU-driven pipeline variants to build.
#[derive(Clone, Copy)]
pub(super) struct MainPipelineFeatures {
    // Build the phase-2 cull PSO + second indirect buffers for two-pass Hi-Z occlusion.
    pub occlusion_two_pass: bool,
    // Build the GPU-driven shadow pass (depth-only pipeline + per-cascade indirect buffers).
    pub shadow_enabled: bool,
    // Build the GPU-driven G-buffer pre-pass (pipeline + previous-frame model buffers).
    pub gbuffer_enabled: bool,
    pub hot_reload: bool,
}

// Build the GPU-driven main pass (root signature, bucket PSOs, GPU-cull compute
// pipeline). Allocates the per-frame `StructuredBuffer<GpuObjectData>` /
// `StructuredBuffer<GpuDrawArgs>` upload buffers and the per-frame
// indirect-command UAV buffers that the cull kernel writes into, all only when
// the world has anything to drive (`n_cull > 0`).
pub(super) fn build_main_pipelines(
    alloc: &DeviceAllocator,
    info_queue: Option<&ID3D12InfoQueue>,
    pipeline_shaders: MainPipelineShaders<'_>,
    config: MainPipelineConfig,
    features: MainPipelineFeatures,
) -> Result<MainPipelines, String> {
    let device = alloc.device();
    let MainPipelineShaders { world_shaders } = pipeline_shaders;
    let world_default = world_shaders
        .first()
        .copied()
        .ok_or_else(|| "BackendInit carried no shaders".to_string())?;
    let bucket_shaders = world_shaders.get(1..).unwrap_or(&[]);
    let MainPipelineConfig {
        n_objects,
        n_instances,
        n_skinned,
        n_chunk_max,
        msaa_samples,
    } = config;
    let MainPipelineFeatures {
        occlusion_two_pass,
        shadow_enabled,
        gbuffer_enabled,
        hot_reload,
    } = features;
    // Merged record count: static build-time objects, the instanced-cluster
    // instances, the runtime reserve, then the skinned objects. The per-frame
    // static fills write only the first `n_objects`; the instance records are
    // written once at init; runtime and skinned records are written each frame
    // into their reserved regions. `n_chunk_max` sizes the streamed-chunk window
    // and `clone_reserve` the spawned-clone one; both live in the single
    // runtime reserve between the instances and the skinned tail (see
    // `DrawState::n_runtime`).
    let n_cull = n_objects
        + n_instances
        + n_chunk_max
        + crate::gfx::render_types::clone_reserve(n_objects)
        + n_skinned;
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
        let max = crate::gfx::render_types::MAX_SHADER_BUCKETS;
        if bucket_shaders.len() + 1 > max {
            return Err(format!(
                "world declares {} Shaders but at most {max} can be routed",
                bucket_shaders.len() + 1
            ));
        }
        build_world_pipeline_table(device, info_queue, targets, bucket_shaders)?
    };
    let main_bindless_root_sig = Some(brs);
    let main_bindless_pso = Some(main_pso);
    let bucket_count = 1 + world_pipelines.len();

    // Per-frame StructuredBuffer<GpuObjectData> upload buffers. Allocated only
    // when the bindless pass is active and the world has build-time static
    // geometry; rebuilt each frame in `build_object_buffer`.
    let mut object_buffer_resources: Vec<PooledBuffer> = Vec::new();
    let mut object_buffer_ptrs: Vec<*mut u8> = Vec::new();
    if main_bindless_pso.is_some() && n_cull > 0 {
        let object_buffer_size = align256(
            (n_cull * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>()) as u64,
        );
        // `FRAMES + 1`: the extra slot (index `FRAMES`) is reserved for the
        // asynchronous reflection-probe capture, which builds its CPU-written
        // bindless buffers into a slot the frame never touches (it uses
        // `[0, FRAMES)`). See `directx/probe.rs::bake_ring_slot`.
        for _ in 0..FRAMES + 1 {
            let buf = create_buffer(
                alloc,
                object_buffer_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map object buffer: {e}"))?;
            object_buffer_ptrs.push(ptr as *mut u8);
            object_buffer_resources.push(buf);
        }
    }

    // Compute cull: cull compute pipeline + per-frame draw-args /
    // indirect-command buffers. Built under the same condition as the object
    // buffer.
    let mut cull_root_sig: Option<ID3D12RootSignature> = None;
    let mut cull_pso: Option<ID3D12PipelineState> = None;
    let mut cull_pso_phase2: Option<ID3D12PipelineState> = None;
    let mut cull_command_signature: Option<ID3D12CommandSignature> = None;
    let mut draw_args_buffer_resources: Vec<PooledBuffer> = Vec::new();
    let mut draw_args_buffer_ptrs: Vec<*mut u8> = Vec::new();
    let mut indirect_cmd_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut cull_status_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut indirect_cmd_buffers_2: Vec<ID3D12Resource> = Vec::new();
    let mut shadow_bindless_root_sig: Option<ID3D12RootSignature> = None;
    let mut shadow_bindless_pso: Option<ID3D12PipelineState> = None;
    let mut shadow_bindless_cmd_sig: Option<ID3D12CommandSignature> = None;
    let mut cull_pso_shadow: Option<ID3D12PipelineState> = None;
    let mut shadow_indirect_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut shadow_cull_status_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut gbuffer_bindless_root_sig: Option<ID3D12RootSignature> = None;
    let mut gbuffer_bindless_pso: Option<ID3D12PipelineState> = None;
    let mut gbuffer_bindless_cmd_sig: Option<ID3D12CommandSignature> = None;
    let mut prev_model_buffer_resources: Vec<PooledBuffer> = Vec::new();
    let mut prev_model_buffer_ptrs: Vec<*mut u8> = Vec::new();
    if let (Some(bindless_root), true) = (
        main_bindless_root_sig.as_ref(),
        main_bindless_pso.is_some() && n_cull > 0,
    ) {
        let cs = compile_cull_shader(hot_reload)?;
        let crs = dump_on_err(info_queue, create_cull_root_signature(device))?;
        let cps = dump_on_err(info_queue, create_cull_pso(device, &crs, &cs))?;
        let csig = dump_on_err(
            info_queue,
            create_cull_command_signature(device, bindless_root),
        )?;
        // Phase-2 cull PSO for two-pass occlusion (same root sig, `main_phase2`
        // entry). Built only when the world opted in.
        if occlusion_two_pass {
            let cs2 = compile_cull_shader_phase2(hot_reload)?;
            cull_pso_phase2 = Some(dump_on_err(
                info_queue,
                create_cull_pso(device, &crs, &cs2),
            )?);
        }

        let draw_args_size = align256(
            (n_cull * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>()) as u64,
        );
        // Default-heap indirect-command buffers (UAV target for the cull
        // kernel; ExecuteIndirect source for the bindless static pass). One
        // `n_cull`-command region per shader bucket: the cull kernel writes every
        // record's slot in each region and the main pass issues one
        // `ExecuteIndirect` per region under that bucket's pipeline.
        let indirect_size =
            align256((bucket_count * n_cull) as u64 * INDIRECT_COMMAND_STRIDE as u64);
        // Per-object cull-status buffer (one u32 each). Always allocated when
        // the cull path is active (matches Metal); resting state `UAV` so it
        // binds as a root UAV with no transition.
        let status_size = align256((n_cull as u64) * std::mem::size_of::<u32>() as u64);
        // `FRAMES + 1`: the extra slot (index `FRAMES`) is the reserved
        // reflection-probe capture slot (see the object-buffer loop above). The
        // bake culls each cube face into `indirect_cmd_buffers[FRAMES]` reading
        // `draw_args_buffer_resources[FRAMES]`, a slot the frame never overwrites.
        for _ in 0..FRAMES + 1 {
            let da = create_buffer(
                alloc,
                draw_args_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { da.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map draw args buffer: {e}"))?;
            draw_args_buffer_ptrs.push(ptr as *mut u8);
            draw_args_buffer_resources.push(da);

            // Created in COMMON (D3D12 always makes committed buffers in COMMON
            // regardless of the requested state); the cull pass transitions them
            // to UNORDERED_ACCESS / INDIRECT_ARGUMENT as it writes + executes them.
            indirect_cmd_buffers.push(create_uav_buffer(
                device,
                indirect_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            cull_status_buffers.push(create_uav_buffer(
                device,
                status_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            // Second indirect buffer for the phase-2 (disocclusion) draws.
            // Only allocated under two-pass occlusion.
            if occlusion_two_pass {
                indirect_cmd_buffers_2.push(create_uav_buffer(
                    device,
                    indirect_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
            }
        }
        // GPU-driven shadow pass: a depth-only bindless pipeline + the shared
        // cull command signature rebuilt against its root sig (object id still
        // at root param 0) + per-frame indirect buffers carrying one cull region
        // per cascade (`NUM_SHADOW_CASCADES * n_cull` commands) + a scratch
        // cull-status buffer the shadow cull dispatches write but never read.
        if shadow_enabled {
            let svs = compile_shadow_bindless_vs(hot_reload)?;
            let sbrs = dump_on_err(info_queue, create_shadow_bindless_root_signature(device))?;
            // Reuse the depth-only shadow PSO builder (no pixel shader, 0 RTVs,
            // D32 DSV, slope-scaled depth bias, main vertex layout).
            let sbpso = dump_on_err(info_queue, create_shadow_pso(device, &sbrs, &svs))?;
            let sbsig = dump_on_err(info_queue, create_cull_command_signature(device, &sbrs))?;
            // Frustum-only shadow cull kernel (`main_shadow`), shares the cull root sig.
            let scs = compile_cull_shader_shadow(hot_reload)?;
            cull_pso_shadow = Some(dump_on_err(
                info_queue,
                create_cull_pso(device, &crs, &scs),
            )?);
            let cascades = crate::gfx::render_types::NUM_SHADOW_CASCADES as u64;
            let shadow_indirect_size =
                align256(cascades * (n_cull as u64) * INDIRECT_COMMAND_STRIDE as u64);
            for _ in 0..FRAMES {
                shadow_indirect_buffers.push(create_uav_buffer(
                    device,
                    shadow_indirect_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
                shadow_cull_status_buffers.push(create_uav_buffer(
                    device,
                    status_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
            }
            shadow_bindless_root_sig = Some(sbrs);
            shadow_bindless_pso = Some(sbpso);
            shadow_bindless_cmd_sig = Some(sbsig);
        }

        // GPU-driven G-buffer pre-pass: a 3-MRT bindless pipeline whose VS reads
        // model + roughness from `GpuObjectData[object_id]` + the previous-frame
        // model from a parallel buffer, drawn by reusing the main pass's per-frame
        // indirect command buffer (NO new cull -- the camera-frustum cull already
        // ran). Plus the per-frame `prev_model` upload buffers (one column-major
        // `float4x4` per cull record): the instance region is init-written, the
        // static + skinned regions rewritten each frame.
        if gbuffer_enabled {
            let (grs, gpso, gsig) = crate::directx::post::gbuffer::build_gbuffer_bindless(
                device, info_queue, hot_reload,
            )?;
            let prev_model_size = align256((n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64);
            for _ in 0..FRAMES {
                let buf = create_buffer(
                    alloc,
                    prev_model_size,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a
                // live local that receives the mapping.
                unsafe { buf.Map(0, None, Some(&mut ptr)) }
                    .map_err(|e| format!("map prev_model buffer: {e}"))?;
                prev_model_buffer_ptrs.push(ptr as *mut u8);
                prev_model_buffer_resources.push(buf);
            }
            gbuffer_bindless_root_sig = Some(grs);
            gbuffer_bindless_pso = Some(gpso);
            gbuffer_bindless_cmd_sig = Some(gsig);
        }

        cull_root_sig = Some(crs);
        cull_pso = Some(cps);
        cull_command_signature = Some(csig);
    }

    Ok(MainPipelines {
        main_bindless_root_sig,
        main_bindless_pso,
        world_pipelines,
        bucket_stride: n_cull,
        bindless_main_shaders,
        object_buffer_resources,
        object_buffer_ptrs,
        cull_root_sig,
        cull_pso,
        cull_pso_phase2,
        cull_command_signature,
        draw_args_buffer_resources,
        draw_args_buffer_ptrs,
        indirect_cmd_buffers,
        cull_status_buffers,
        indirect_cmd_buffers_2,
        shadow_bindless_root_sig,
        shadow_bindless_pso,
        shadow_bindless_cmd_sig,
        cull_pso_shadow,
        shadow_indirect_buffers,
        shadow_cull_status_buffers,
        gbuffer_bindless_root_sig,
        gbuffer_bindless_pso,
        gbuffer_bindless_cmd_sig,
        prev_model_buffer_resources,
        prev_model_buffer_ptrs,
    })
}

pub(super) fn build_shadow_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    shadow_vs: Option<&[u8]>,
) -> Result<(Option<ID3D12RootSignature>, Option<ID3D12PipelineState>), String> {
    if let Some(svs) = shadow_vs {
        let sr = dump_on_err(info_queue, create_shadow_root_signature(device))?;
        let sp = dump_on_err(info_queue, create_shadow_pso(device, &sr, svs))?;
        Ok((Some(sr), Some(sp)))
    } else {
        Ok((None, None))
    }
}

pub(super) fn build_text_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    text_vs: &[u8],
    text_ps: &[u8],
    swap_format: DXGI_FORMAT,
    has_atlases: bool,
) -> Result<(ID3D12RootSignature, Option<ID3D12PipelineState>), String> {
    let text_root_sig = dump_on_err(info_queue, create_text_root_signature(device))?;
    // Text renders in the composite pass into the single-sample swapchain
    // backbuffer (post-tonemap), so its PSO targets the swapchain format at
    // sample count 1.
    let text_pso = if has_atlases {
        Some(dump_on_err(
            info_queue,
            create_text_pso(device, &text_root_sig, text_vs, text_ps, swap_format, 1),
        )?)
    } else {
        None
    };
    Ok((text_root_sig, text_pso))
}

pub(super) fn build_composite_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    swap_format: DXGI_FORMAT,
    hot_reload: bool,
) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    let composite_root_sig = dump_on_err(info_queue, create_composite_root_signature(device))?;
    let (composite_vs, composite_ps) = compile_composite_shaders(hot_reload)?;
    let composite_pso = dump_on_err(
        info_queue,
        create_composite_pso(
            device,
            &composite_root_sig,
            &composite_vs,
            &composite_ps,
            swap_format,
        ),
    )?;
    Ok((composite_root_sig, composite_pso))
}

#[cfg(test)]
mod tests {
    // The bindless main pair compiles from `src/shaders/main_bindless.slang` at
    // runtime (slangc, DXIL sm 6.0). This compiles it offline so a syntax or
    // register error fails a test instead of only surfacing as an init failure
    // on a GPU host.
    #[test]
    fn bindless_main_shaders_compile() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        super::compile_main_bindless_shaders(false).expect("bindless main shaders must compile");
    }
}

// src/directx/init/pipelines.rs
//
// Core render-pipeline construction extracted from DxContext::new:
//   * Shader compilation (`compile_shaders`, `compile_main_bindless_shaders`),
//     from the built-in program declarations in `directx/builtins.rs`.
//   * Root-signature + PSO builders for the main pass, the GPU-cull bindless
//     variant, the GPU-instanced main pass, and the depth-only shadow pass.
//   * High-level `build_main_pipelines`/`build_shadow_pipeline`/etc.
//     orchestration helpers consumed by init/mod.rs.
//
// Mirrors src/metal/init/pipelines.rs (the same set of pipelines built at
// init time). Text + composite pipelines live in `directx/pipeline.rs`;
// bloom/TAA/SSAO live in `directx/post/`; the GPU-cull compute pipeline lives
// in `directx/cull.rs`; the skinned-mesh pipelines (built lazily once a
// `SkinnedMesh` is uploaded) live in `directx/resources.rs`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtins;
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
use crate::directx::texture::{HDR_FORMAT, create_buffer, create_uav_buffer};

// Shader compilation

pub(super) struct CompiledShaders {
    pub main_vs: Vec<u8>,
    pub main_ps: Vec<u8>,
    pub shadow_vs: Option<Vec<u8>>,
    pub main_vs_instanced: Option<Vec<u8>>,
    pub text_vs: Vec<u8>,
    pub text_ps: Vec<u8>,
}

// Compile every shader stage the init path needs. `vert_bytes`, `frag_bytes`,
// `vert_instanced_bytes`, and `shadow_bytes` are pre-compiled DXBC overrides
// when non-empty (matching the metallib override model on Metal). `shadow_vs`
// is `None` when no shadow shader is configured.
pub(super) fn compile_all_shaders(
    vert_bytes: &[u8],
    frag_bytes: &[u8],
    shadow_bytes: &[u8],
    vert_instanced_bytes: &[u8],
    need_instanced: bool,
    hot_reload: bool,
) -> Result<CompiledShaders, String> {
    let main_vs = if !vert_bytes.is_empty() {
        vert_bytes.to_vec()
    } else {
        builtins::MAIN_VERT.compile(hot_reload)?
    };
    let main_ps = if !frag_bytes.is_empty() {
        frag_bytes.to_vec()
    } else {
        builtins::MAIN_FRAG.compile(hot_reload)?
    };
    // The shadow vertex shader is engine-internal: a real DXBC override (>4
    // bytes) is used verbatim, otherwise (empty / stub) the baked
    // `slang_builtins::SHADOW_VERT` is compiled. Whether the shadow pass runs is
    // gated by `effective_shadow_size` at the call site, not by an empty
    // override here.
    let shadow_vs = if shadow_bytes.len() > 4 {
        Some(shadow_bytes.to_vec())
    } else {
        Some(slang_builtins::SHADOW_VERT.compile(hot_reload)?)
    };
    let main_vs_instanced = if !vert_instanced_bytes.is_empty() {
        Some(vert_instanced_bytes.to_vec())
    } else if need_instanced {
        Some(builtins::MAIN_VERT_INSTANCED.compile(hot_reload)?)
    } else {
        None
    };
    let (text_vs, text_ps) = compile_text_shaders(hot_reload)?;
    Ok(CompiledShaders {
        main_vs,
        main_ps,
        shadow_vs,
        main_vs_instanced,
        text_vs,
        text_ps,
    })
}

// Compile the bindless static-pass shaders (bindless static pass). Always built
// from the single-source `.slang` pair; the bindless path only ever drives the
// built-in shader, and worlds that supply a custom main shader either keep the
// legacy pipeline or bring their own bucket PSO.
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

fn create_main_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    // Descriptor ranges for tables.
    // [4] table layout (SRVs at heap slots 0..3 inclusive):
    //   range 1: shadow_map_array at t0    (heap slot 0)
    //   range 2: irradiance + prefilter cubes at t5..t6 (heap slots 1..2)
    // Both ranges use APPEND so the runtime places them back-to-back from the
    // table base; matches the heap layout in context.rs.
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
    let object_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // t1..t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // s0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [7] table covers linear repeat sampler (s1) + cube sampler (s2)
    // contiguous in the sampler heap.
    let linear_cube_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // s1..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [8] table: SSAO occlusion SRV at t4 (or a 1x1 white fallback so the
    // shader's ambient *= ssao_tex.r is a pass-through when SSAO is off).
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [13] table: spot shadow depth array at t10 (a 1x1 fallback array when the
    // world has no shadow-casting spot).
    let spot_shadow_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 10, // t10
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [15] table: the two area-light LTC tables at t12..t13, contiguous in the
    // heap so one range covers both.
    let ltc_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 12, // t12..t13
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constants: model mat4 + material = 28 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 28,
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
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
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
        // [5] Descriptor table: albedo + normal SRVs (t1..t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &object_srv_range,
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
        // [8] Descriptor table: SSAO occlusion SRV (t4)
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
        // [9] Root SRV: per-scene StructuredBuffer<GpuLight> at t7 (matches
        // main_frag.hlsl; t3 is the instanced/skinned VS matrix SRV in the
        // shared instanced root signature, so the lights sit at t7).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 7,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [10] Root CBV: ClusterParams at b4 (clustered lighting).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 4,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [11] Root SRV: per-cluster light-index lists at t8.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 8,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [12] Root SRV: per-slice StructuredBuffer<SpotShadowData> at t9.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 9,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [13] table: spot shadow depth array at t10. A texture cannot be a root
        // descriptor, so unlike the buffer above it needs a table.
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
        // [14] Root SRV: per-scene StructuredBuffer<AreaLightData> at t11.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 11,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [15] table: the area-light LTC tables at t12..t13.
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

    serialize_and_create_root_sig(device, &params, "main root sig")
}

// Root signature for the bindless static main pass (bindless static pass).
//
// Differs from `create_main_root_signature`: slot [0] is a single-DWORD root
// constant carrying just the per-draw object id (D3D12 `SV_InstanceID` does
// not include `StartInstanceLocation`, so the id rides a root constant); slot
// [5] is the unbounded bindless `Texture2D` pool (`t0, space1`) instead of the
// per-object albedo/normal table; slot [8] is a root SRV at `t3` carrying the
// per-frame `StructuredBuffer<GpuObjectData>`. The per-object descriptor table
// is gone; that was the per-draw binding the compute-driven cull
// needed removed.
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
    // [9] table: SSAO occlusion SRV at t4 (same convention as the legacy
    // main root sig; the same bindless fragment shader samples it).
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
        NumDescriptors: concinnity_render::uniforms::MAX_PROBES as u32,
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
        BaseShaderRegister: 7 + concinnity_render::uniforms::MAX_PROBES as u32 + 1, // t16
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [18] table: the area-light LTC tables, past the spot shadow array.
    let ltc_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 7 + concinnity_render::uniforms::MAX_PROBES as u32 + 3, // t18..t19
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
                    ShaderRegister: 7 + concinnity_render::uniforms::MAX_PROBES as u32, // t15
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
                    ShaderRegister: 7 + concinnity_render::uniforms::MAX_PROBES as u32 + 2, // t17,
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

// Same as the main root signature but with one extra root SRV at slot [8]
// (t3) carrying per-instance world matrices. Used by the GPU-instanced PSO
// and also the skinned PSO (whose root SRV at the same slot carries joint
// matrices instead).
pub(in crate::directx) fn create_main_instanced_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let shadow_srv_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0, // t0 shadow_map_array
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 5, // t5..t6 IBL cubes
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    let object_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 1,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
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
    // [9] table: SSAO occlusion SRV at t4 (matches main + bindless layout).
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [14] table: spot shadow depth array at t10 (matches the main layout).
    let spot_shadow_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 10, // t10
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [16] table: the area-light LTC tables at t12..t13 (matches the main layout).
    let ltc_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 12, // t12..t13
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constants at b0 (same as main; model field is ignored by VS)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 28,
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
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [4] Descriptor table: shadow array (t0) + IBL cubes (t5..t6)
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
        // [5] Descriptor table: albedo + normal SRVs (t1..t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &object_srv_range,
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
        // [8] Root SRV: per-instance world matrices (t3, VS-only)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
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
        // [10] Root SRV: per-scene StructuredBuffer<GpuLight> at t7 (PS). The VS
        // matrix / joint SRV holds t3, so the local lights sit at t7 to match
        // main_frag.hlsl and the static main root signature.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 7,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [11] Root CBV: ClusterParams at b4 (clustered lighting).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 4,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [12] Root SRV: per-cluster light-index lists at t8.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 8,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [13] Root SRV: per-slice StructuredBuffer<SpotShadowData> at t9.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 9,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [14] table: spot shadow depth array at t10.
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
        // [15] Root SRV: per-scene StructuredBuffer<AreaLightData> at t11.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 11,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [16] table: the area-light LTC tables at t12..t13.
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

    serialize_and_create_root_sig(device, &params, "main instanced root sig")
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

// PSO for the main (static + instanced + bindless) pass. The instanced
// pipeline reuses this with the appropriate VS + root sig.
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
// pipeline state (unlike Metal's encoder flag), so the mode needs its own PSO
// per main-pass pipeline; see [`super::super::wireframe`].
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
            DepthBias: 1,
            DepthBiasClamp: 0.01,
            SlopeScaledDepthBias: 1.0,
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
// without recompiling the HLSL. Recompiling cost ~140 ms per bucket install,
// which is the whole point of warming a pipeline behind a loading screen.
pub(in crate::directx) struct BindlessMainShaders {
    pub vs: Vec<u8>,
    pub ps: Vec<u8>,
}

// Build one shader bucket's bindless main-pass pipeline. `bucket` is the
// `DrawObject::shader_bucket` value (1-based; bucket 0 is the world default
// program) and names the bucket in error messages.
//
// Empty `vert` bytes mean the world declared no Shader for this bucket, so the
// engine's own bindless program renders it.
pub(in crate::directx) fn build_bucket_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    bindless_root_sig: &ID3D12RootSignature,
    bucket: usize,
    shader: crate::gfx::backend_init::ShaderBytes<'_>,
    msaa_samples: u32,
    engine_default: &BindlessMainShaders,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = if shader.vert.is_empty() {
        (engine_default.vs.as_slice(), engine_default.ps.as_slice())
    } else {
        (shader.vert, shader.frag)
    };
    if vs.is_empty() || ps.is_empty() {
        return Err(format!(
            "shader bucket {bucket} carries no vertex/fragment bytecode"
        ));
    }
    dump_on_err(
        info_queue,
        create_main_pso(device, bindless_root_sig, vs, ps, HDR_FORMAT, msaa_samples),
    )
    .map_err(|e| format!("shader bucket {bucket}: {e}"))
}

// Build the per-bucket pipeline table from the world's material-referenced
// shaders. Index `b` holds bucket `b + 1`'s pipeline; `None` marks a bucket the
// streaming pump installs later (its Shader is owned by a scene that has not
// pinned, so `decode_shaders` handed over an all-empty payload).
fn build_world_pipeline_table(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    bindless_root_sig: &ID3D12RootSignature,
    bucket_shaders: &[crate::gfx::backend_init::ShaderBytes<'_>],
    msaa_samples: u32,
    engine_default: &BindlessMainShaders,
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
            device,
            info_queue,
            bindless_root_sig,
            bucket,
            *shader,
            msaa_samples,
            engine_default,
        )?));
    }
    Ok(table)
}

// Init-time orchestration

pub(super) struct MainPipelines {
    pub main_root_sig: ID3D12RootSignature,
    pub main_pso: ID3D12PipelineState,
    pub main_bindless_root_sig: Option<ID3D12RootSignature>,
    pub main_bindless_pso: Option<ID3D12PipelineState>,
    // Material-referenced world shader pipelines, indexed by `shader_bucket - 1`.
    // Empty unless the world declares more than one Shader.
    pub world_pipelines: Vec<Option<ID3D12PipelineState>>,
    // Commands reserved per bucket region in the indirect buffers.
    pub bucket_stride: usize,
    // The engine's compiled bindless main-pass stages, retained for the buckets a
    // scene warms mid-session. Empty when the world authored its own main shader
    // (there is no bindless path to warm into).
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

// Build the main static pass + (when the world ships no custom main shader)
// the bindless variant and GPU-cull compute pipeline. Allocates the per-frame
// `StructuredBuffer<GpuObjectData>` / `StructuredBuffer<GpuDrawArgs>` upload
// buffers and the per-frame indirect-command UAV buffers that the cull kernel
// writes into.
//
// The bindless + cull infrastructure is only built when
// `vert_bytes`+`frag_bytes` are empty (built-in shader path) AND
// `n_objects > 0`. Otherwise the corresponding fields are `None` / empty.
// Compiled + optional override shader bytes for the main pass pipelines.
#[derive(Clone, Copy)]
pub(super) struct MainPipelineShaders<'a> {
    // Built-in compiled main pass shaders.
    pub shaders: &'a CompiledShaders,
    // Precompiled DXBC override for a custom vertex shader (empty = use built-in).
    pub vert_bytes: &'a [u8],
    // Precompiled DXBC override for a custom fragment shader (empty = use built-in).
    pub frag_bytes: &'a [u8],
    // The world's material-referenced shaders (`BackendInit::shaders[1..]`), one
    // per shader bucket past the default. Each gets its own bindless main-pass
    // pipeline; an entry flagged `deferred` is a bucket whose Shader belongs to
    // a scene that has not pinned, and is installed later by
    // `install_world_shader`.
    pub bucket_shaders: &'a [crate::gfx::backend_init::ShaderBytes<'a>],
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

pub(super) fn build_main_pipelines(
    alloc: &DeviceAllocator,
    info_queue: Option<&ID3D12InfoQueue>,
    pipeline_shaders: MainPipelineShaders<'_>,
    config: MainPipelineConfig,
    features: MainPipelineFeatures,
) -> Result<MainPipelines, String> {
    let device = alloc.device();
    let MainPipelineShaders {
        shaders,
        vert_bytes,
        frag_bytes,
        bucket_shaders,
    } = pipeline_shaders;
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
    // instances, the streamed-chunk reserve, then the skinned objects. The
    // per-frame static fills write only the first `n_objects`; the instance records
    // are written once at init; chunk records + skinned records are written each
    // frame into their reserved regions.
    let n_cull = n_objects + n_instances + n_chunk_max + n_skinned;
    let main_root_sig = dump_on_err(info_queue, create_main_root_signature(device))?;
    let main_pso = dump_on_err(
        info_queue,
        create_main_pso(
            device,
            &main_root_sig,
            &shaders.main_vs,
            &shaders.main_ps,
            HDR_FORMAT,
            msaa_samples,
        ),
    )?;

    // Bindless static main pass (bindless static pass). Built only when no custom
    // main shader was supplied; a world with its own shader keeps the legacy
    // per-draw pipeline.
    let main_is_builtin = vert_bytes.is_empty() && frag_bytes.is_empty();
    let mut bindless_main_shaders = BindlessMainShaders {
        vs: Vec::new(),
        ps: Vec::new(),
    };
    let (main_bindless_root_sig, main_bindless_pso) = if main_is_builtin {
        let (bvs, bps) = compile_main_bindless_shaders(hot_reload)?;
        let brs = dump_on_err(info_queue, create_main_bindless_root_signature(device))?;
        let bpso = dump_on_err(
            info_queue,
            create_main_pso(device, &brs, &bvs, &bps, HDR_FORMAT, msaa_samples),
        )?;
        bindless_main_shaders = BindlessMainShaders { vs: bvs, ps: bps };
        (Some(brs), Some(bpso))
    } else {
        (None, None)
    };

    // Material-referenced shaders (ShaderHandle 1..) each get their own bindless
    // main-pass pipeline, so their draws route into their own region of the
    // GPU-culled command buffer. They exist only on the bindless path: a world
    // with a legacy per-draw main shader carries no bucket routing at all.
    let world_pipelines = match main_bindless_root_sig.as_ref() {
        Some(brs) if !bucket_shaders.is_empty() => {
            let max = crate::gfx::render_types::MAX_SHADER_BUCKETS;
            if bucket_shaders.len() + 1 > max {
                return Err(format!(
                    "world declares {} Shaders but at most {max} can be routed",
                    bucket_shaders.len() + 1
                ));
            }
            build_world_pipeline_table(
                device,
                info_queue,
                brs,
                bucket_shaders,
                msaa_samples,
                &bindless_main_shaders,
            )?
        }
        _ if !bucket_shaders.is_empty() => {
            return Err(
                "material-referenced world shaders need the bindless main pass, which a \
                 world-authored main shader disables"
                    .to_string(),
            );
        }
        _ => Vec::new(),
    };
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
        main_root_sig,
        main_pso,
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

pub(super) fn build_main_instanced_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    instanced_vs: Option<&[u8]>,
    main_ps: &[u8],
    msaa_samples: u32,
) -> Result<(Option<ID3D12RootSignature>, Option<ID3D12PipelineState>), String> {
    if let Some(ivs) = instanced_vs {
        let irs = dump_on_err(info_queue, create_main_instanced_root_signature(device))?;
        let ips = dump_on_err(
            info_queue,
            create_main_pso(device, &irs, ivs, main_ps, HDR_FORMAT, msaa_samples),
        )?;
        Ok((Some(irs), Some(ips)))
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

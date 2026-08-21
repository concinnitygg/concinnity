// src/directx/post/ssr.rs
//
// Screen-space reflections for the D3D12 backend: a fullscreen ray-march
// resolve that reads the resolved HDR scene plus the unified G-buffer pre-pass
// (view normal + linear depth + roughness, from post/gbuffer.rs), ray-marches
// the reflection, and composites it into the SSR output target. The output then
// replaces the raw HDR resolve as the "scene" colour the TAA / bloom / composite
// passes consume.
//
// Mirrors src/metal/post/ssr.rs.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};
use crate::gfx::render_types::SsrParams;

use crate::directx::com;
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::serialize_desc_and_create;
use crate::directx::post::gbuffer::GbufferResources;
use crate::directx::slang_builtins;
use crate::directx::texture::{
    create_buffer, create_rt_target, write_format_rtv, write_format_srv,
};

// HDR-format SSR resolve output. Replaces `hdr_resolve` as the scene colour
// the TAA / bloom / composite passes consume when SSR is on.
pub const SSR_OUTPUT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Size of the SSR resolve fragment-shader uniform block. 96 bytes; see
// `gfx::render_types::SsrParams`.
const SSR_PARAMS_UBO_SIZE: u64 = 96;

// First register of the reflection-probe cube array, and the register of the one
// sampler it shares. slangc numbers resources from declaration order, so this is
// the count of screen sources `ssr.slang` declares ahead of the array.
const PROBE_CUBE_REGISTER: u32 = 4;

// Shader compilation

struct SsrShaders {
    resolve_vs: Vec<u8>,
    resolve_ps: Vec<u8>,
}

// Compile the SSR resolve shader stages. The resolve fullscreen pass has no
// geometry input; the view normal / depth / roughness come from the unified
// G-buffer pre-pass.
fn compile_ssr_shaders(hot_reload: bool) -> Result<SsrShaders, String> {
    Ok(SsrShaders {
        resolve_vs: slang_builtins::FULLSCREEN_VERT.compile(hot_reload)?,
        resolve_ps: slang_builtins::SSR_RESOLVE.compile(hot_reload)?,
    })
}

// Root signatures

// Root signature for the fullscreen SSR resolve: a root CBV at b0 (the 96-byte
// `SsrParams` block) and four 1-SRV descriptor tables: scene (t0), G-buffer
// normal+depth (t1), roughness (t2), and the IBL prefilter cubemap (t3), then
// the reflection-probe cube array at t4.. with its `ProbeSet` at b1. One static
// sampler per texture the shader declares (s0..s3 plus s4.. for the probe
// array), because slangc splits each combined sampler in `ssr.slang` into its
// own texture/sampler pair.
fn create_ssr_resolve_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let scene_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let gbuffer_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let rough_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 2, // t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let cube_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 3, // t3
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [5] table: the reflection-probe cube array at t4..t4+MAX_PROBES, the next
    // free register after the four screen sources `ssr.slang` declares ahead of
    // it. Unbaked slots hold the sky prefilter cube, so a sample at any index is
    // valid; the miss fallback box-projects these when ProbeSet.count > 0.
    let probe_cube_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: concinnity_render::uniforms::MAX_PROBES as u32,
        BaseShaderRegister: PROBE_CUBE_REGISTER,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Root CBV: SsrParams at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [1] scene SRV (t0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &scene_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [2] gbuffer SRV (t1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &gbuffer_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] roughness SRV (t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &rough_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [4] prefilter cubemap SRV (t3)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &cube_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [5] reflection-probe cube array table (t4..)
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
        // [6] Root CBV: ProbeSet at b1, the second constant buffer `ssr.slang`
        // declares after the params push.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    // One sampler per declared texture, all linear-clamp with mip-linear
    // filtering: s0..s2 for the screen sources, s3 for the prefilter cube, and
    // s4 shared by the whole probe cube array (which the shader declares as a
    // texture array plus that one sampler, because D3D12 binds a sampler array
    // only through a descriptor table). Identical descriptors; the per-texture
    // split is the shader's, not the pass's.
    let linear_clamp = |reg: u32| D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: reg,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    // s0..s3 plus the probe array's shared s4.
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> =
        (0..=PROBE_CUBE_REGISTER).map(linear_clamp).collect();
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "ssr resolve root sig")
}

// PSO builders

// PSO for the fullscreen SSR resolve pass. Writes `SSR_OUTPUT_FORMAT`; no
// depth + no blending.
fn create_ssr_resolve_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
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
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = SSR_OUTPUT_FORMAT;
            a
        },
        DSVFormat: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: false.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
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
        .map_err(|e| format!("create ssr resolve PSO: {e}"))
}

// Resources

// SSR resolve state. Held by `SsrResources::resolve` and `Some` only when the
// SSR resolve is actually authored on. SSGI reuses the unified G-buffer pre-pass
// without needing the resolve, so a world that enables SSGI but not SSR leaves
// this `None`.
pub(in crate::directx) struct SsrResolve {
    // Resolved authored tunables; turned into a per-frame `SsrParams` push.
    pub(in crate::directx) settings: crate::gfx::ssr::SsrSettings,

    // SSR resolve output: the HDR scene with reflections composited in.
    // Becomes the "scene" SRV the TAA / bloom / composite passes consume.
    pub(in crate::directx) output: ID3D12Resource,
    pub(in crate::directx) output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Per-frame resolve params UBO (96-byte SsrParams), persistently mapped.
    pub(in crate::directx) params_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,

    // Resolve fullscreen pass.
    pub(in crate::directx) resolve_root_sig: ID3D12RootSignature,
    pub(in crate::directx) resolve_pso: ID3D12PipelineState,
}

// SSR resources held by `DxContext` when `PostProcessConfig.ssr` is on, or when
// SSGI is on (both reuse the unified G-buffer pre-pass). The resolve half is
// `Some` only when the SSR resolve itself is authored on. Drops cleanly with the
// context: all D3D12 objects are COM-refcounted.
pub(in crate::directx) struct SsrResources {
    // SSR resolve (fullscreen ray-march). `None` for a SSGI-only build, which
    // reuses the unified G-buffer without writing a replacement scene.
    pub(in crate::directx) resolve: Option<SsrResolve>,
}

// Output target descriptors + resolve settings for the SSR resolve builder.
pub(in crate::directx) struct SsrInitInputs {
    // Resolved authored tunables; `Some` only when the SSR resolve is on. With
    // `None` (a SSGI-only build) the resolve output + pipeline are skipped.
    pub resolve_settings: Option<crate::gfx::ssr::SsrSettings>,
    // SSR resolve output: CPU RTV plus the (CPU, GPU) SRV pair TAA / bloom /
    // composite consume as the scene colour.
    pub output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

impl SsrResources {
    // Build the SSR resolve resources. Called from `DxContext::new` when the
    // world's `PostProcessConfig` enables SSR or SSGI. `resolve_settings` is
    // `Some` only when the SSR resolve itself is authored on; with `None` (a
    // SSGI-only build) the resolve output, its params UBO, and its pipeline are
    // skipped and the reserved output slot goes unwritten.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        width: u32,
        height: u32,
        inputs: SsrInitInputs,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let device = alloc.device();
        let SsrInitInputs {
            resolve_settings,
            output_rtv,
            output_srv,
        } = inputs;
        let resolve = if let Some(settings) = resolve_settings {
            let output = create_rt_target(device, width, height, SSR_OUTPUT_FORMAT)?;
            write_format_rtv(device, &output, output_rtv, SSR_OUTPUT_FORMAT);
            write_format_srv(device, &output, output_srv.0, SSR_OUTPUT_FORMAT);

            let params_size = align256(SSR_PARAMS_UBO_SIZE);
            let mut params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
            let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
            for _ in 0..FRAMES {
                let buf = create_buffer(
                    alloc,
                    params_size,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a
                // live local that receives the mapping.
                unsafe { buf.Map(0, None, Some(&mut ptr)) }
                    .map_err(|e| format!("map ssr params ubo: {e}"))?;
                params_ubo_ptrs.push(ptr as *mut u8);
                params_ubo_resources.push(buf);
            }

            let shaders = compile_ssr_shaders(hot_reload)?;
            let resolve_root_sig =
                dump_on_err(info_queue, create_ssr_resolve_root_signature(device))?;
            let resolve_pso = dump_on_err(
                info_queue,
                create_ssr_resolve_pso(
                    device,
                    &resolve_root_sig,
                    &shaders.resolve_vs,
                    &shaders.resolve_ps,
                ),
            )?;
            Some(SsrResolve {
                settings,
                output,
                output_rtv,
                output_srv_gpu: output_srv.1,
                params_ubo_resources,
                params_ubo_ptrs,
                resolve_root_sig,
                resolve_pso,
            })
        } else {
            None
        };

        Ok(Self { resolve })
    }
}

// Replacement SSR PSO returned by [`rebuild_ssr_pipelines`]. `None` when the SSR
// resolve is off (SSGI reusing the G-buffer only).
pub(in crate::directx) struct RebuiltSsrPipelines {
    pub resolve_pso: Option<ID3D12PipelineState>,
}

impl SsrResources {
    // Rebuild the SSR resolve output at a new resolution. The descriptor *slot*
    // stays where it was; only the resource backing it changes. The "scene SRV"
    // the post stack consumes when SSR is on is `output_srv_gpu`, so this
    // rewrite is enough for the composite chain to keep working.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<(), String> {
        let srv_cpu = |gpu: D3D12_GPU_DESCRIPTOR_HANDLE| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + (gpu.ptr - srv_gpu_base.ptr) as usize,
        };

        if let Some(r) = self.resolve.as_mut() {
            r.output = create_rt_target(device, width, height, SSR_OUTPUT_FORMAT)?;
            write_format_rtv(device, &r.output, r.output_rtv, SSR_OUTPUT_FORMAT);
            write_format_srv(
                device,
                &r.output,
                srv_cpu(r.output_srv_gpu),
                SSR_OUTPUT_FORMAT,
            );
        }

        Ok(())
    }
}

// Rebuild the SSR resolve PSO against fresh shader source. Reuses the existing
// root signature; returns the new PSO for the caller to swap into the live
// `SsrResources`.
pub(in crate::directx) fn rebuild_ssr_pipelines(
    device: &ID3D12Device,
    ssr: &SsrResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltSsrPipelines, String> {
    let resolve_pso = if let Some(r) = ssr.resolve.as_ref() {
        let shaders = compile_ssr_shaders(hot_reload)?;
        Some(dump_on_err(
            info_queue,
            create_ssr_resolve_pso(
                device,
                &r.resolve_root_sig,
                &shaders.resolve_vs,
                &shaders.resolve_ps,
            ),
        )?)
    } else {
        None
    };
    Ok(RebuiltSsrPipelines { resolve_pso })
}

// Encoders

impl DxContext {
    // GPU descriptor handle of the SRV the TAA / bloom / composite passes
    // sample as the "scene": the FSR3 upscale output when temporal
    // upscaling is on, the SSR resolve output when SSR is on (and
    // upscale is off), otherwise the raw `hdr_resolve` SRV. The upscale
    // branch wins because FSR consumes the post-SSR scene and produces
    // the final temporally-accumulated image; the post stack reads from
    // that.
    // The scene texture the post stack consumes *before* the upscaler: the
    // reflection composite's output when a resolve ran, otherwise the HDR spine.
    // These two are the graph's `scene_pre_taa` and `hdr_resolve`, and this
    // predicate is the one the graph builder branches on.
    //
    // It exists because `scene_srv_for_post` below picks the same texture and
    // the two drifted apart once: when the SSR / RT resolves stopped
    // compositing inline and started writing radiance + weight into their own
    // target, the upscaler kept reading that target as if it were the scene, so
    // a world with both SSR and temporal upscaling upscaled the reflection
    // buffer. Anything that needs the resource rather than its SRV goes through
    // here.
    pub(in crate::directx) fn post_scene_target(&self) -> &ID3D12Resource {
        match self
            .reflection_composite
            .as_ref()
            .filter(|_| self.reflection_resolve_active())
        {
            Some(rc) => &rc.output,
            None => self.hdr_scene_target(),
        }
    }

    // Render-target view of `post_scene_target`, for the passes that blend into
    // it rather than sample it.
    pub(in crate::directx) fn post_scene_rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        match self
            .reflection_composite
            .as_ref()
            .filter(|_| self.reflection_resolve_active())
        {
            Some(rc) => rc.output_rtv,
            None => self.hdr_scene_rtv(),
        }
    }

    pub(in crate::directx) fn scene_srv_for_post(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        if let Some(up) = &self.upscale.backend {
            return up.output_srv_gpu();
        }
        // When a reflection resolve ran this frame, the roughness composite wrote the
        // scene-with-reflections into its own output; the post stack samples that.
        // Both SSR and RT feed the same composite (RT takes precedence at the graph
        // level, so at most one resolve runs). A SSGI-only world runs no resolve, so
        // it samples `hdr_resolve` directly (SSGI already composited its bounce in).
        if let Some(rc) = self.reflection_composite.as_ref()
            && self.reflection_resolve_active()
        {
            return rc.output_srv_gpu;
        }
        self.hdr.srv_gpu
    }

    // GPU descriptor handle of the IBL prefilter cubemap SRV. Fixed at heap
    // slot 2; the SSR resolve and RT-reflection resolve both bind it as a miss
    // fallback. With no `EnvironmentMap` declared, the slot holds a 1×1 grey
    // fallback cube and `prefilter_mip_count == 0` tells the resolve to skip it.
    pub(in crate::directx) fn prefilter_cube_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (2 * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // Encode the SSR resolve: a fullscreen triangle that ray-marches the
    // reflection through `hdr_resolve` (already in `PIXEL_SHADER_RESOURCE`
    // state after the main pass) and composites the result into
    // `ssr.resolve.output`. The output then becomes the "scene" the TAA /
    // bloom / composite passes consume via `scene_srv_for_post`. No-op when the
    // resolve half is absent (a SSGI-only build) or the G-buffer is missing.
    pub(in crate::directx) fn encode_ssr_resolve(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        // Resolve the pass's resources up front (a SSGI-only build leaves the
        // resolve half absent; the G-buffer is always built when SSR is on).
        // Resolving here, before the encoder, keeps the driver from ever leaving
        // the barrier bracket half-open.
        let Some(ssr) = &self.ssr else { return };
        let Some(resolve) = &ssr.resolve else { return };
        let Some(gbuffer) = &self.gbuffer else { return };
        // The resolve writes reflected radiance + weight into `resolve.output`.
        let reflection_srv = resolve.output_srv_gpu;
        encode_fullscreen(
            &SsrResolvePass {
                ctx: self,
                resolve,
                gbuffer,
                frame_idx,
                fov_y_radians,
                aspect,
                cam_pos,
            },
            cmd,
        );
        // Blur the reflection by roughness and composite it over the scene into the
        // reflection-composite output (the scene the post stack then consumes).
        self.encode_reflection_composite(cmd, reflection_srv);
    }
}

// Encoder for the SSR resolve fullscreen pass: the resolved resources + the
// per-call view params. The PSR<->RENDER_TARGET barrier bracket + render-target
// bind live in `DxContext::begin/end_fullscreen_rt` (post/fullscreen.rs); only
// the SSR-specific bind + draw is here. Constructed + driven by
// `encode_ssr_resolve` through `gfx::fullscreen::encode_fullscreen`.
struct SsrResolvePass<'a> {
    ctx: &'a DxContext,
    resolve: &'a SsrResolve,
    gbuffer: &'a GbufferResources,
    frame_idx: usize,
    fov_y_radians: f32,
    aspect: f32,
    cam_pos: [f32; 3],
}

impl FullscreenPass for SsrResolvePass<'_> {
    type Rec = ID3D12GraphicsCommandList;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx
            .begin_fullscreen_rt(cmd, &self.resolve.output, self.resolve.output_rtv);
    }

    fn draw(&self, cmd: &Self::Rec) {
        // Build the per-frame SsrParams: turn the view matrix's 3x3 into the
        // view→world rotation (its transpose, embedded in a 4x4), then upload.
        let v = self.ctx.view.matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = self.resolve.settings.params(
            self.fov_y_radians,
            self.aspect,
            inv_view_rot,
            self.cam_pos,
            self.ctx.env_map.prefilter_mip_count as f32,
        );
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const SsrParams as *const u8,
                self.resolve.params_ubo_ptrs[self.frame_idx],
                std::mem::size_of::<SsrParams>(),
            );
        }
        let params_gva = com::gpu_va(&self.resolve.params_ubo_resources[self.frame_idx]);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(&self.resolve.resolve_pso);
            cmd.SetGraphicsRootSignature(&self.resolve.resolve_root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, params_gva);
            cmd.SetGraphicsRootDescriptorTable(1, self.ctx.hdr.srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(2, self.gbuffer.normal_depth_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(3, self.gbuffer.roughness_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(4, self.ctx.prefilter_cube_srv_gpu());
            // Reflection-probe miss fallback: the cube array table
            // at t7 + the per-frame ProbeSet CBV at b4. count == 0 keeps the exact sky
            // path, so a probe-less world is byte-identical to before.
            cmd.SetGraphicsRootDescriptorTable(5, self.ctx.probe_cube_table_gpu());
            cmd.SetGraphicsRootConstantBufferView(
                6,
                com::gpu_va(&self.ctx.probe.set_cbvs[self.frame_idx]),
            );
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_rt(cmd, &self.resolve.output);
    }
}

#[cfg(test)]
mod tests {
    // The resolve fragment compiles from `ssr.slang` at renderer init. Compile it
    // offline so a source or register error in the reflection-probe miss fallback
    // fails a test instead of only surfacing as an init failure on the GPU host.
    // Skipped on a host without slangc, matching `concinnity_slang`'s own
    // round-trip tests.
    #[test]
    fn ssr_resolve_shaders_compile() {
        if concinnity_slang::slangc_path().is_none() {
            return;
        }
        super::compile_ssr_shaders(false).expect("ssr resolve shaders must compile");
    }
}

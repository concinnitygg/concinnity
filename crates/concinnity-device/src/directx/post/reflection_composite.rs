// src/directx/post/reflection_composite.rs
//
// Roughness-aware reflection composite for the D3D12 backend. The SSR and RT
// resolves now write reflected radiance (.rgb) + a Fresnel/gloss weight (.a) into
// their output target instead of compositing inline; this two-pass effect blurs
// that reflection by surface roughness and composites it over the scene into a
// shared output the TAA / bloom / composite passes consume.
//
//   reflection_blur      (half-res): weight-averages the reflection over a
//       roughness-scaled cone into `blur`. The expensive multi-tap part, run at a
//       fraction of the pixels.
//   reflection_composite (full-res): lerps the sharp full-res reflection against
//       the upsampled half-res blur by roughness, then composites over the scene.
//
// Shared by both reflection paths: `encode_ssr_resolve` / `encode_rt_reflections`
// each render their reflection target, then call `encode_reflection_composite` with
// that target's SRV. Mirrors src/metal/post/ssr.rs (the composite half).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{DxContext, dump_on_err};
use crate::directx::pipeline::serialize_desc_and_create;
use crate::directx::post::ssr::SSR_OUTPUT_FORMAT;
use crate::directx::slang_builtins;
use crate::directx::texture::{create_rt_target, write_format_rtv, write_format_srv};

// The blur pass runs at render-resolution / `blur_scale`. The blur is low-
// frequency (a widening glossy cone), so running it reduced and bilinear-
// upsampling in the composite is visually free while cutting its pixel count;
// mirrors stay sharp (the composite lerps in the full-res reflection for low
// roughness). The divisor is resolved per world from
// `PostProcessConfig.reflection_blur_resolution` (Half=2 default, matching the
// historical hardcoded scale) and stored on the resources so resize reuses it.

// Shader compilation

struct ReflCompShaders {
    vs: Vec<u8>,
    blur_ps: Vec<u8>,
    composite_ps: Vec<u8>,
}

// Compile the shared fullscreen vertex shader + the blur + composite fragment
// entry points. `REFLECTION_ROUGHNESS_CUT` is a `static const` in
// `reflection.slang`, locked to the canonical Rust value by unit test, so the
// blur ramp matches the SSR / RT resolve gates.
fn compile_refl_composite_shaders(hot_reload: bool) -> Result<ReflCompShaders, String> {
    Ok(ReflCompShaders {
        vs: slang_builtins::FULLSCREEN_VERT.compile(hot_reload)?,
        blur_ps: slang_builtins::REFLECTION_BLUR.compile(hot_reload)?,
        composite_ps: slang_builtins::REFLECTION_COMPOSITE.compile(hot_reload)?,
    })
}

// Root signatures

// A descriptor table of `count` consecutive SRVs starting at register t0, plus
// one static sampler per SRV. Both passes index their inputs t0.. as APPEND
// ranges; the samplers are s0..s(count-1) because slangc splits each combined
// sampler in the single source into its own texture/sampler pair.
fn srv_table_root_sig(
    device: &ID3D12Device,
    count: u32,
    name: &str,
) -> Result<ID3D12RootSignature, String> {
    let ranges: Vec<D3D12_DESCRIPTOR_RANGE> = (0..count)
        .map(|i| D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: i,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        })
        .collect();
    let params: Vec<D3D12_ROOT_PARAMETER> = ranges
        .iter()
        .map(|r| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: r,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        })
        .collect();
    // One linear-clamp sampler per input (reflection / scene / G-buffer / blur).
    // Identical descriptors; the split is the shader's, not the pass's.
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> = (0..count)
        .map(|reg| D3D12_STATIC_SAMPLER_DESC {
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
        })
        .collect();
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, name)
}

// Fullscreen PSO writing `SSR_OUTPUT_FORMAT` (RGBA16F): no depth, no blend, no
// vertex input.
fn create_fullscreen_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef (see the SSR PSO builder).
        // SAFETY: a raw pointer copy with no refcount change; the borrowed COM object outlives the
        // call, and the `ManuallyDrop` field never releases it.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
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
        .map_err(|e| format!("create reflection composite PSO: {e}"))
}

// Resources

// Reflection-composite resources, held by `DxContext` when SSR resolve OR RT
// reflections are authored (both feed the same composite). The `output` is the
// scene-with-reflections the post stack consumes via `scene_srv_for_post`.
pub(in crate::directx) struct ReflectionCompositeResources {
    // Composited scene (full render resolution): the blurred reflection over the
    // scene. Becomes the scene colour TAA / bloom / composite / glass consume.
    pub(in crate::directx) output: ID3D12Resource,
    pub(in crate::directx) output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Reduced-resolution roughness blur of the reflection target (the blur pass
    // writes it, the composite upsamples it). Sized at render / `blur_scale`.
    blur: ID3D12Resource,
    blur_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    blur_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Per-axis divisor the blur target is sized by, resolved from the world's
    // `reflection_blur_resolution`. Held so `resize_to` reuses the same scale.
    blur_scale: u32,

    blur_root_sig: ID3D12RootSignature,
    blur_pso: ID3D12PipelineState,
    composite_root_sig: ID3D12RootSignature,
    composite_pso: ID3D12PipelineState,
}

// Slot handles minted by the caller (mod.rs owns the heap layout). Copy so the
// caller can both build the resources at init and stash a copy in
// `QualitySlotHandles` for a live `apply_quality_settings` reflection enable.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ReflectionCompositeSlots {
    pub output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub blur_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub blur_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

impl ReflectionCompositeResources {
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        blur_scale: u32,
        slots: ReflectionCompositeSlots,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let blur_scale = blur_scale.max(1);
        let output = create_rt_target(device, width, height, SSR_OUTPUT_FORMAT)?;
        write_format_rtv(device, &output, slots.output_rtv, SSR_OUTPUT_FORMAT);
        write_format_srv(device, &output, slots.output_srv.0, SSR_OUTPUT_FORMAT);

        let bw = (width / blur_scale).max(1);
        let bh = (height / blur_scale).max(1);
        let blur = create_rt_target(device, bw, bh, SSR_OUTPUT_FORMAT)?;
        write_format_rtv(device, &blur, slots.blur_rtv, SSR_OUTPUT_FORMAT);
        write_format_srv(device, &blur, slots.blur_srv.0, SSR_OUTPUT_FORMAT);

        let shaders = compile_refl_composite_shaders(hot_reload)?;
        // Blur reads reflection (t0) + roughness (t1); the composite declares
        // them in its own order -- reflection (t0), scene (t1), G-buffer
        // normal+depth (t2), roughness (t3), blur (t4).
        let blur_root_sig = dump_on_err(
            info_queue,
            srv_table_root_sig(device, 2, "reflection blur root sig"),
        )?;
        let composite_root_sig = dump_on_err(
            info_queue,
            srv_table_root_sig(device, 5, "reflection composite root sig"),
        )?;
        let blur_pso = dump_on_err(
            info_queue,
            create_fullscreen_pso(device, &blur_root_sig, &shaders.vs, &shaders.blur_ps),
        )?;
        let composite_pso = dump_on_err(
            info_queue,
            create_fullscreen_pso(
                device,
                &composite_root_sig,
                &shaders.vs,
                &shaders.composite_ps,
            ),
        )?;

        Ok(Self {
            output,
            output_rtv: slots.output_rtv,
            output_srv_gpu: slots.output_srv.1,
            blur,
            blur_rtv: slots.blur_rtv,
            blur_srv_gpu: slots.blur_srv.1,
            blur_scale,
            blur_root_sig,
            blur_pso,
            composite_root_sig,
            composite_pso,
        })
    }

    // Recreate the output + blur targets at a new resolution, rewriting their SRVs
    // in place (the descriptor slots do not move, so the post stack's scene binding
    // stays valid). Mirrors `SsrResources::resize_to`.
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
        self.output = create_rt_target(device, width, height, SSR_OUTPUT_FORMAT)?;
        write_format_rtv(device, &self.output, self.output_rtv, SSR_OUTPUT_FORMAT);
        write_format_srv(
            device,
            &self.output,
            srv_cpu(self.output_srv_gpu),
            SSR_OUTPUT_FORMAT,
        );

        let bw = (width / self.blur_scale).max(1);
        let bh = (height / self.blur_scale).max(1);
        self.blur = create_rt_target(device, bw, bh, SSR_OUTPUT_FORMAT)?;
        write_format_rtv(device, &self.blur, self.blur_rtv, SSR_OUTPUT_FORMAT);
        write_format_srv(
            device,
            &self.blur,
            srv_cpu(self.blur_srv_gpu),
            SSR_OUTPUT_FORMAT,
        );
        Ok(())
    }
}

// Rebuilt PSOs from a shader hot-reload; swapped into the live resources.
pub(in crate::directx) struct RebuiltReflectionComposite {
    pub blur_pso: ID3D12PipelineState,
    pub composite_pso: ID3D12PipelineState,
}

pub(in crate::directx) fn rebuild_reflection_composite_pipelines(
    device: &ID3D12Device,
    rc: &ReflectionCompositeResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltReflectionComposite, String> {
    let shaders = compile_refl_composite_shaders(hot_reload)?;
    let blur_pso = dump_on_err(
        info_queue,
        create_fullscreen_pso(device, &rc.blur_root_sig, &shaders.vs, &shaders.blur_ps),
    )?;
    let composite_pso = dump_on_err(
        info_queue,
        create_fullscreen_pso(
            device,
            &rc.composite_root_sig,
            &shaders.vs,
            &shaders.composite_ps,
        ),
    )?;
    Ok(RebuiltReflectionComposite {
        blur_pso,
        composite_pso,
    })
}

pub(in crate::directx) fn swap_reflection_composite_pipelines(
    rc: &mut ReflectionCompositeResources,
    rebuilt: RebuiltReflectionComposite,
) {
    rc.blur_pso = rebuilt.blur_pso;
    rc.composite_pso = rebuilt.composite_pso;
}

// Encoder

impl DxContext {
    // Blur the reflection target by surface roughness and composite it over the
    // scene into `reflection_composite.output`. `reflection_srv` is the SRV of the
    // resolve target the SSR / RT pass just wrote (reflected radiance + weight); it
    // rests in PIXEL_SHADER_RESOURCE after the resolve. No-op when the composite or
    // G-buffer is absent (only when no reflection path is active).
    pub(in crate::directx) fn encode_reflection_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        reflection_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        let Some(rc) = &self.reflection_composite else {
            return;
        };
        let Some(gbuffer) = &self.gbuffer else {
            return;
        };

        // Pass 1: the roughness blur into the reduced-resolution `blur` target
        // (begin_fullscreen_rt sizes the viewport to the target's own dimensions).
        self.begin_fullscreen_rt(cmd, &rc.blur, rc.blur_rtv);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(&rc.blur_pso);
            cmd.SetGraphicsRootSignature(&rc.blur_root_sig);
            cmd.SetGraphicsRootDescriptorTable(0, reflection_srv);
            cmd.SetGraphicsRootDescriptorTable(1, gbuffer.roughness_srv_gpu);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        self.end_fullscreen_rt(cmd, &rc.blur);

        // Pass 2: lerp the sharp full-res reflection against the upsampled blur by
        // roughness, then composite over the scene into `output` -- the graph's
        // `scene_pre_taa`, which the executor has already put in RENDER_TARGET
        // for this pass's declared write, and which the next consumer's barrier
        // takes back out.
        self.bind_fullscreen_rt(cmd, &rc.output, rc.output_rtv);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(&rc.composite_pso);
            cmd.SetGraphicsRootSignature(&rc.composite_root_sig);
            // t0 reflection, t1 scene, t2 G-buffer normal+depth, t3 roughness,
            // t4 blur -- the order `reflection.slang`'s composite declares them.
            cmd.SetGraphicsRootDescriptorTable(0, reflection_srv);
            cmd.SetGraphicsRootDescriptorTable(1, self.hdr.srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(2, gbuffer.normal_depth_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(3, gbuffer.roughness_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(4, rc.blur_srv_gpu);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    // The blur + composite fragments compile from `reflection.slang` at renderer
    // init. Compile them offline so a source or register error fails a test
    // instead of only an init failure on the GPU host. Skipped on a host without
    // slangc, matching `concinnity_slang`'s own round-trip tests.
    #[test]
    fn reflection_composite_shaders_compile() {
        if concinnity_slang::slangc_path().is_none() {
            return;
        }
        super::compile_refl_composite_shaders(false)
            .expect("reflection composite shaders must compile");
    }
}

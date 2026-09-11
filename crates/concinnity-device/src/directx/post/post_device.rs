// src/directx/post/post_device.rs
//
// DirectX's implementation of the shared fullscreen post-pass seam
// (`gfx::post::PostPassDevice`).
//
// The root signature is derived from what the single source declares rather
// than hand-written per pass: N single-SRV descriptor tables at t0..tN-1, a
// 32-bit constant block at b0 when the program declares constants, and N static
// samplers at s0..sN-1. The tables are single-SRV and separate because slangc
// numbers each top-level `Sampler2D` into its own texture and sampler slot from
// declaration order, and because a pass's sources come from unrelated owners, so
// nothing makes them contiguous in the heap.
//
// Targets take their SRV and RTV from the shared post descriptor block
// (post/descriptors.rs) instead of slots reserved for the effect by name in
// `init/heap_layout.rs`.

use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, resolved_texture,
};
use concinnity_core::render::post::program::PostProgram;
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::dump_on_err;
use crate::directx::pipeline::{create_blended_composite_pso, serialize_desc_and_create};
use crate::directx::post::descriptors::{PostDescriptors, PostTargetDescriptors};
use crate::directx::post::fullscreen::FullscreenExtent;
use crate::directx::slang_builtins::{self, SlangCompile};
use crate::directx::texture::{create_rt_target, write_format_rtv, write_format_srv};
use crate::directx::transient_pool::dxgi_format;

// A built fullscreen post pipeline: the PSO, the root signature it was derived
// against, and the counts that derivation used, so a draw can check what it was
// handed.
pub(in crate::directx) struct PostPipeline {
    pub(in crate::directx) pso: ID3D12PipelineState,
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    textures: usize,
    constants: usize,
}

// A persistent post target: the resource, its descriptors, and the extent it was
// created at.
pub(in crate::directx) struct PostTarget {
    pub(in crate::directx) resource: ID3D12Resource,
    pub(in crate::directx) descriptors: PostTargetDescriptors,
    extent: FullscreenExtent,
}

impl PostTarget {
    // The shader-visible handle a consumer samples this target through.
    pub(in crate::directx) fn srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.descriptors.srv_gpu
    }
}

// The D3D12 handles a shared post pass builds and encodes through.
pub(in crate::directx) struct DxPostDevice<'a> {
    pub device: &'a ID3D12Device,
    // Where a target's SRV and RTV come from.
    pub descriptors: &'a PostDescriptors,
    // The shader-visible heap a draw's root tables index, bound per pass.
    pub srv_heap: &'a ID3D12DescriptorHeap,
    // The debug-layer message queue, so a root-signature or PSO failure prints
    // what the layer said rather than just an HRESULT.
    pub info_queue: Option<&'a ID3D12InfoQueue>,
    pub hot_reload: bool,
}

// The DXIL a post program's two stages compile to: the one shared fullscreen
// triangle vertex plus the program's own fragment.
fn compile(program: PostProgram, hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let frag = match program {
        PostProgram::TaaResolve => &slang_builtins::TAA_FRAG,
    };
    Ok((
        slang_builtins::FULLSCREEN_VERT.compile(hot_reload)?,
        frag.compile(hot_reload)?,
    ))
}

// A root signature for `textures` single-SRV tables plus, when the program
// declares constants, a 32-bit constant block at b0. Static linear clamp-to-edge
// samplers at s0..sN-1: each source's sampler is its own slot because slangc
// splits every combined `Sampler2D` into a texture and a sampler at the same
// index.
fn create_root_signature(
    device: &ID3D12Device,
    textures: usize,
    constants: usize,
) -> Result<ID3D12RootSignature, String> {
    let ranges: Vec<D3D12_DESCRIPTOR_RANGE> = (0..textures as u32)
        .map(|reg| D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: reg,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        })
        .collect();
    let mut params: Vec<D3D12_ROOT_PARAMETER> = ranges
        .iter()
        .map(|range| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        })
        .collect();
    if constants > 0 {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: constants.div_ceil(4) as u32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        });
    }
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> = (0..textures as u32)
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
    serialize_desc_and_create(device, &desc, "post root sig")
}

impl PostPassDevice for DxPostDevice<'_> {
    type Recorder = ID3D12GraphicsCommandList;
    type Pipeline = PostPipeline;
    type Target = PostTarget;
    type TextureRef<'a> = D3D12_GPU_DESCRIPTOR_HANDLE;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> Result<Self::Pipeline, String> {
        let bindings = program.bindings();
        let root_sig = dump_on_err(
            self.info_queue,
            create_root_signature(self.device, bindings.textures, bindings.constants),
        )?;
        let (vs, ps) = compile(program, self.hot_reload)?;
        let pso = dump_on_err(
            self.info_queue,
            create_blended_composite_pso(
                self.device,
                &root_sig,
                &vs,
                &ps,
                dxgi_format(format),
                blend,
                program.label(),
            ),
        )?;
        Ok(PostPipeline {
            pso,
            root_sig,
            textures: bindings.textures,
            constants: bindings.constants,
        })
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> Result<Self::Target, String> {
        let spec = resolved_texture(label, desc, extent);
        let format = dxgi_format(spec.format);
        let resource = create_rt_target(self.device, spec.width, spec.height, format)?;
        let descriptors = self.descriptors.allocate()?;
        write_format_srv(self.device, &resource, descriptors.srv_cpu, format);
        write_format_rtv(self.device, &resource, descriptors.rtv, format);
        Ok(PostTarget {
            resource,
            descriptors,
            extent: FullscreenExtent {
                width: spec.width,
                height: spec.height,
            },
        })
    }

    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a> {
        target.descriptors.srv_gpu
    }

    fn encode(&self, cmd: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> Result<(), String> {
        let pipe = draw.pipeline;
        if draw.binds.len() != pipe.textures || draw.constants.len() != pipe.constants {
            return Err(format!(
                "{}: the draw binds {} texture(s) and {} constant byte(s) where the program \
                 declares {} and {}",
                draw.label,
                draw.binds.len(),
                draw.constants.len(),
                pipe.textures,
                pipe.constants,
            ));
        }
        // The render graph drives every post target, so the executor has already
        // put this one in RENDER_TARGET; the next consumer's graph barrier takes
        // it back out. The load action is therefore the pass's own business and
        // there is nothing to clear: the fullscreen triangle covers every pixel.
        debug_assert!(matches!(draw.load, PostLoadOp::DontCare | PostLoadOp::Load));
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&draw.target.descriptors.rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: draw.target.extent.width as f32,
                Height: draw.target.extent.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            cmd.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: draw.target.extent.width as i32,
                bottom: draw.target.extent.height as i32,
            }]);
            cmd.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            cmd.SetPipelineState(&pipe.pso);
            cmd.SetGraphicsRootSignature(&pipe.root_sig);
            // Every source binds through a static sampler the root signature
            // declared at its own slot, so a bind is just its table.
            for (slot, bind) in draw.binds.iter().enumerate() {
                cmd.SetGraphicsRootDescriptorTable(slot as u32, bind.texture);
            }
            if !draw.constants.is_empty() {
                cmd.SetGraphicsRoot32BitConstants(
                    pipe.textures as u32,
                    draw.constants.len().div_ceil(4) as u32,
                    draw.constants.as_ptr() as *const std::ffi::c_void,
                    0,
                );
            }
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            // The vertex stage builds the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        Ok(())
    }
}

impl crate::directx::context::DxContext {
    // The post-pass device over this context.
    pub(in crate::directx) fn post_device(&self) -> DxPostDevice<'_> {
        DxPostDevice {
            device: &self.device,
            descriptors: &self.post,
            srv_heap: &self.descriptors.srv_heap,
            info_queue: self.diagnostics.info_queue.as_ref(),
            hot_reload: self.hot_reload.enabled,
        }
    }
}

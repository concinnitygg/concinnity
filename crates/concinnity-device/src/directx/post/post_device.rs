// src/directx/post/post_device.rs
//
// DirectX's implementation of the shared fullscreen post-pass seam
// (`render::post::device::PostPassDevice`).
//
// The root signature is derived from what the single source declares rather
// than hand-written per pass: N single-SRV descriptor tables at t0..tN-1, a
// 32-bit constant block at b0 when the program declares constants, and N static
// samplers at s0..sN-1. A probe-reading program adds the reflection-probe cube
// array at tN.. with its one static sampler at sN, and the ProbeSet at b1. The
// tables are single-SRV and separate because slangc numbers each top-level
// combined sampler into its own texture and sampler slot from declaration
// order, and because a pass's sources come from unrelated owners, so nothing
// makes them contiguous in the heap.
//
// Targets take their SRV and RTV from the shared post descriptor block
// (post/descriptors.rs) instead of slots reserved for the effect by name in
// `init/heap_layout.rs`.

use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostTargetState, resolved_texture,
};
use concinnity_core::render::post::program::{PostProgram, PostProgramBindings};
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};
use concinnity_core::render::uniforms::MAX_PROBES;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::com;
use crate::directx::context::dump_on_err;
use crate::directx::pipeline::{create_blended_composite_pso, serialize_desc_and_create};
use crate::directx::post::descriptors::{PostDescriptors, PostTargetDescriptors};
use crate::directx::post::fullscreen::FullscreenExtent;
use crate::directx::slang_builtins::{self, SlangCompile};
use crate::directx::texture::{
    create_rt_target, transition_barrier, write_format_rtv, write_format_srv,
};
use crate::directx::transient_pool::dxgi_format;

// The register a probe-reading program declares its ProbeSet at, after the
// constants at b0.
const PROBE_SET_REGISTER: u32 = 1;

// A built fullscreen post pipeline: the PSO, the root signature it was derived
// against, and the declaration that derivation used, so a draw can check what
// it was handed and find its root parameters.
pub(in crate::directx) struct PostPipeline {
    pub(in crate::directx) pso: ID3D12PipelineState,
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    bindings: PostProgramBindings,
}

impl PostPipeline {
    // Root parameter of the constant block, which follows the source tables.
    fn constants_parameter(&self) -> u32 {
        self.bindings.textures as u32
    }

    // Root parameter of the probe cube table; the ProbeSet is the one after it.
    fn probes_parameter(&self) -> u32 {
        self.bindings.textures as u32 + u32::from(self.bindings.constants > 0)
    }
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

// A draw's color target: the resource (for the transitions of a target its pass
// owns), the RTV it is written through, and its extent, which is the viewport.
#[derive(Clone, Copy)]
pub(in crate::directx) struct DxAttachment<'a> {
    pub resource: &'a ID3D12Resource,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub extent: FullscreenExtent,
}

// The world's reflection-probe set, as a probe-reading program binds it.
#[derive(Clone, Copy)]
pub(in crate::directx) struct DxPostProbes {
    // The `MAX_PROBES` contiguous cube SRVs.
    pub cube_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    // This frame's ProbeSet constant buffer.
    pub set_cbv: u64,
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
    // The probe set a probe-reading program binds. Absent at init, where no
    // draw is encoded.
    pub probes: Option<DxPostProbes>,
    pub hot_reload: bool,
}

// The DXIL a post program's two stages compile to: the one shared fullscreen
// triangle vertex plus the program's own fragment.
fn compile(program: PostProgram, hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let frag = match program {
        PostProgram::TaaResolve => &slang_builtins::TAA_FRAG,
        PostProgram::SsrResolve => &slang_builtins::SSR_RESOLVE,
        PostProgram::SsgiGather => &slang_builtins::SSGI_GATHER,
        PostProgram::SsgiComposite => &slang_builtins::SSGI_COMPOSITE,
    };
    Ok((
        slang_builtins::FULLSCREEN_VERT.compile(hot_reload)?,
        frag.compile(hot_reload)?,
    ))
}

fn srv_range(register: u32, count: u32) -> D3D12_DESCRIPTOR_RANGE {
    D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: count,
        BaseShaderRegister: register,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    }
}

fn table_parameter(range: &D3D12_DESCRIPTOR_RANGE) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    }
}

// A root signature for `bindings`: one single-SRV table per source, a 32-bit
// constant block at b0 when the program declares constants, and for a
// probe-reading program the cube array table plus a root CBV for the ProbeSet.
// Static samplers at s0..sN-1, plus sN for the cube array: every sampler kind
// the seam names resolves to the same trilinear clamp-to-edge state here, so the
// signature does not depend on which kind a draw picks for a slot.
fn create_root_signature(
    device: &ID3D12Device,
    bindings: PostProgramBindings,
) -> Result<ID3D12RootSignature, String> {
    let textures = bindings.textures as u32;
    let ranges: Vec<D3D12_DESCRIPTOR_RANGE> = (0..textures).map(|reg| srv_range(reg, 1)).collect();
    // The cube array takes the next texture register after the declared sources.
    let probe_range = srv_range(textures, MAX_PROBES as u32);
    let mut params: Vec<D3D12_ROOT_PARAMETER> = ranges.iter().map(table_parameter).collect();
    if bindings.constants > 0 {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: bindings.constants.div_ceil(4) as u32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        });
    }
    if bindings.probes {
        params.push(table_parameter(&probe_range));
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: PROBE_SET_REGISTER,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        });
    }
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> = (0..textures + u32::from(bindings.probes))
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
    type Attachment<'a> = DxAttachment<'a>;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> Result<Self::Pipeline, String> {
        let bindings = program.bindings();
        let root_sig = dump_on_err(
            self.info_queue,
            create_root_signature(self.device, bindings),
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
            bindings,
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

    fn target_attachment<'a>(&self, target: &'a Self::Target) -> Self::Attachment<'a> {
        DxAttachment {
            resource: &target.resource,
            rtv: target.descriptors.rtv,
            extent: target.extent,
        }
    }

    fn encode(&self, cmd: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> Result<(), String> {
        let pipe = draw.pipeline;
        draw.check(pipe.bindings)?;
        let probes = match (pipe.bindings.probes, self.probes) {
            (false, _) => None,
            (true, Some(probes)) => Some(probes),
            (true, None) => {
                return Err(format!(
                    "{}: the program reads the reflection-probe set, but this device holds none",
                    draw.label
                ));
            }
        };
        let target = draw.target;
        // A target the graph declares is already in RENDER_TARGET and its next
        // consumer's barrier takes it back out. One private to its pass rests
        // readable, so the draw moves it itself. Either way there is nothing to
        // clear: the fullscreen triangle covers every pixel, and a load keeps
        // what the target holds.
        let owns_state = draw.state == PostTargetState::Pass;
        debug_assert!(matches!(draw.load, PostLoadOp::DontCare | PostLoadOp::Load));
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            if owns_state {
                cmd.ResourceBarrier(&[transition_barrier(
                    target.resource,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
            cmd.OMSetRenderTargets(1, Some(&target.rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: target.extent.width as f32,
                Height: target.extent.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            cmd.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: target.extent.width as i32,
                bottom: target.extent.height as i32,
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
                    pipe.constants_parameter(),
                    draw.constants.len().div_ceil(4) as u32,
                    draw.constants.as_ptr() as *const std::ffi::c_void,
                    0,
                );
            }
            if let Some(probes) = probes {
                cmd.SetGraphicsRootDescriptorTable(pipe.probes_parameter(), probes.cube_table);
                cmd.SetGraphicsRootConstantBufferView(pipe.probes_parameter() + 1, probes.set_cbv);
            }
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            // The vertex stage builds the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
            if owns_state {
                cmd.ResourceBarrier(&[transition_barrier(
                    target.resource,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        }
        Ok(())
    }
}

impl crate::directx::context::DxContext {
    // The post-pass device over this context, binding frame slot `frame`'s
    // reflection-probe set.
    pub(in crate::directx) fn post_device(&self, frame: usize) -> DxPostDevice<'_> {
        DxPostDevice {
            device: &self.device,
            descriptors: &self.post,
            srv_heap: &self.descriptors.srv_heap,
            info_queue: self.diagnostics.info_queue.as_ref(),
            probes: Some(DxPostProbes {
                cube_table: self.probe_cube_table_gpu(),
                set_cbv: com::gpu_va(&self.probe.set_cbvs[frame]),
            }),
            hot_reload: self.hot_reload.enabled,
        }
    }

    // The single-sample HDR scene the graph drives as `hdr_resolve`, as a post
    // draw's target.
    pub(in crate::directx) fn hdr_scene_attachment(&self) -> DxAttachment<'_> {
        DxAttachment {
            resource: self.hdr_scene_target(),
            rtv: self.hdr_scene_rtv(),
            extent: FullscreenExtent {
                width: self.extent.render_width,
                height: self.extent.render_height,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every post program's fragment, and the shared vertex, compile to DXIL, so a
    // source or register error fails a test instead of only surfacing as an
    // init failure on the GPU host. Skipped on a host without slangc.
    #[test]
    fn every_post_program_compiles() {
        if !concinnity_slang::shader_tests_enabled() {
            return;
        }
        for program in [
            PostProgram::TaaResolve,
            PostProgram::SsrResolve,
            PostProgram::SsgiGather,
            PostProgram::SsgiComposite,
        ] {
            compile(program, false).unwrap_or_else(|e| panic!("{program:?}: {e}"));
        }
    }
}

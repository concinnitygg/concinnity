// src/directx/line.rs
//
// World-space line pass for the D3D12 backend. Runs at the tail of the
// hdr_resolve decoration chain, after the main pass resolved colour into the
// HDR scene target and depth into the main depth buffer, so the lines layer
// over the lit scene and SSR / TAA treat them like any other scene content.
//
// The ribbons arrive already expanded (`gfx::lines::build_vertices`):
// world-space quads whose width was sized off each corner's depth, so a line
// holds its pixel thickness at any distance. Like the decal pass this one
// attaches no depth buffer and instead samples the scene depth, so an occluded
// line fades to `OCCLUDED_ALPHA` rather than being clipped by hardware.
//
// Mirrors src/metal/line.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::serialize_desc_and_create;
use crate::directx::slang_builtins;
use crate::directx::texture::{HDR_FORMAT, create_buffer};
use crate::directx::upload_ring::{UPLOAD_ALIGN, UploadRing, align_up};
use crate::gfx::render_types::LineVertex;

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the lines readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// `LineView` is a GPU-free layout struct that lives in concinnity-render;
// re-export it so `crate::directx::line::LineView` is the local path.
pub(in crate::directx) use concinnity_render::uniforms::LineView;

// Line-pass state on the context: the resources, built on the first frame that
// submits lines so a world that never draws any pays nothing, plus the
// build-failure latch that keeps a broken build from re-reporting every frame.
pub(in crate::directx) struct LineState {
    pub resources: Option<LineResources>,
    pub build_failed: bool,
}

impl LineState {
    pub(in crate::directx) fn empty() -> Self {
        Self {
            resources: None,
            build_failed: false,
        }
    }
}

// Owned by `DxContext` at most once (built lazily): the line pipeline, the
// per-frame view CBV ring, and the per-frame ribbon-vertex upload ring. Nothing
// here is sized off the render target, so a swapchain resize leaves it intact.
pub(in crate::directx) struct LineResources {
    root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,

    // Per-frame view CBV (single 80-byte block), persistently mapped.
    view_ubo_resources: Vec<PooledBuffer>,
    view_ubo_ptrs: Vec<*mut u8>,

    // Per-frame ribbon vertices. Sized to the frame's expanded line set.
    vertices: UploadRing,

    // Heap slot of the main-depth SRV, bound at t0; the resource is
    // transitioned to PIXEL_SHADER_RESOURCE around the pass.
    depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

impl LineResources {
    fn new(
        alloc: &DeviceAllocator,
        msaa_samples: u32,
        depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let device = alloc.device();
        let (vs, ps) = compile_line_shaders(msaa_samples, hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_line_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_line_pso(device, &root_sig, &vs, &ps))?;

        let view_size = align256(std::mem::size_of::<LineView>() as u64);
        let mut view_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                alloc,
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map line view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        Ok(Self {
            root_sig,
            pso,
            view_ubo_resources,
            view_ubo_ptrs,
            vertices: UploadRing::new(FRAMES),
            depth_srv_gpu,
        })
    }
}

// Compile the line vertex + fragment shaders; the MSAA variant keeps the
// fragment shader's depth SRV declaration in sync with the resource's sample
// count. Used by the lazy build and by shader hot-reload.
fn compile_line_shaders(msaa_samples: u32, hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let frag = if msaa_samples > 1 {
        &slang_builtins::LINE_FRAG_MSAA
    } else {
        &slang_builtins::LINE_FRAG
    };
    let vs = slang_builtins::LINE_VERT.compile(hot_reload)?;
    let ps = frag.compile(hot_reload)?;
    Ok((vs, ps))
}

// Rebuild the line PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; the root signature is reused.
pub(in crate::directx) fn rebuild_line_pso(
    device: &ID3D12Device,
    lines: &LineResources,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_line_shaders(msaa_samples, hot_reload)?;
    dump_on_err(
        info_queue,
        create_line_pso(device, &lines.root_sig, &vs, &ps),
    )
}

// Root-signature layout (binds 1:1 with the `line.slang` declarations, whose
// registers slangc assigns from declaration order; `SLANG_DXIL_ENTRY_ABI` in
// build.rs pins each one):
//   [0] root CBV b0   LineView (per-frame)
//   [1] table  t0     scene depth SRV (Texture2D[MS]<float>)
// No sampler: the fragment shader `Load`s the depth texel under the pixel.
fn create_line_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let depth_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &depth_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    serialize_desc_and_create(device, &desc, "line root sig")
}

// Vertex input elements for the line pass (32-byte `LineVertex` struct),
// asserted by `line_vertex_layout_matches_shaders`.
fn line_input_layout() -> [D3D12_INPUT_ELEMENT_DESC; 3] {
    [
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 16,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// PSO for the line pass: world-space ribbon corners transformed by the camera
// VP and alpha-blended into the resolved HDR target. No depth attachment; the
// fragment shader tests the scene depth itself so an occluded line can fade
// instead of vanishing. No culling either: a ribbon faces the camera but its
// winding depends on which way the line runs.
fn create_line_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = line_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
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
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = HDR_FORMAT;
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
                    BlendEnable: true.into(),
                    SrcBlend: D3D12_BLEND_SRC_ALPHA,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_SRC_ALPHA,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
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
        .map_err(|e| format!("create line PSO: {e}"))
}

// Encoder

impl DxContext {
    // Build the line resources if this frame has lines to draw and they are
    // not built yet. A failed build latches, so the error is reported once and
    // the pass stays skipped for the rest of the run.
    pub(in crate::directx) fn ensure_line_pipeline(&mut self, has_lines: bool) {
        if !has_lines || self.lines.resources.is_some() || self.lines.build_failed {
            return;
        }
        let info_queue = self.info_queue.clone();
        match LineResources::new(
            &self.alloc,
            self.hdr.msaa_samples,
            self.main_depth_srv_gpu,
            info_queue.as_ref(),
            self.hot_reload.enabled,
        ) {
            Ok(r) => self.lines.resources = Some(r),
            Err(e) => {
                self.lines.build_failed = true;
                tracing::error!("line pipeline: {}", e);
            }
        }
    }

    // Encode the line pass: one unindexed triangle list covering every expanded
    // ribbon, alpha-blended into the resolved HDR target. `vp` is the same
    // view-projection the main pass rasterised with (jittered under TAA), so a
    // line sits on the pixel its geometry did.
    pub(in crate::directx) fn encode_lines(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        vertices: &[LineVertex],
    ) -> Result<(), String> {
        let Some(lines) = self.lines.resources.as_ref() else {
            return Ok(());
        };
        if vertices.is_empty() {
            return Ok(());
        }

        let view_uni = LineView {
            vp,
            occluded_alpha: OCCLUDED_ALPHA,
            _pad: [0.0; 3],
        };
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for this payload, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const LineView as *const u8,
                lines.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<LineView>(),
            );
        }
        // SAFETY: a property query on a live resource; it only reads.
        let view_gva = unsafe { lines.view_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Ribbon vertices into this frame's slot of the upload ring. The frame
        // fence (waited before the slot is reused) already retired the lists
        // that read it last trip.
        let vertex_bytes: &[u8] = bytemuck::cast_slice(vertices);
        lines.vertices.reserve(
            &self.alloc,
            frame_idx,
            align_up(vertex_bytes.len() as u64, UPLOAD_ALIGN),
        )?;
        let vertex_gva = lines.vertices.push(frame_idx, vertex_bytes)?;
        let vertex_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vertex_gva,
            SizeInBytes: vertex_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<LineVertex>() as u32,
        };

        // Main depth is already in a shader-resource state for the fragment's
        // occlusion sample: the graph declares this pass's depth read and the
        // executor emits the transition ahead of this command list.

        // The scene spine is a graph resource: this pass declares its
        // read-modify-write, so the executor has already put it in RENDER_TARGET
        // and the next consumer's barrier takes it back out.
        let scene_rtv = self.hdr_scene_rtv();

        let w = self.render_width;
        let h = self.render_height;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&scene_rtv), false, None);
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            }]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[vertex_view]));

            cmd.SetPipelineState(&lines.pso);
            cmd.SetGraphicsRootSignature(&lines.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootDescriptorTable(1, lines.depth_srv_gpu);
            cmd.DrawInstanced(vertices.len() as u32, 1, 0, 0);
        }
        self.inc_draw_calls(1);
        Ok(())
    }
}

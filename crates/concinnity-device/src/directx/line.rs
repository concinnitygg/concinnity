//! World-space line pass for the D3D12 backend. Runs at the tail of the
//! hdr_resolve decoration chain, after the main pass resolved color into the
//! HDR scene target and depth into the main depth buffer, so the lines layer
//! over the lit scene and SSR / TAA treat them like any other scene content.
//!
//! The ribbons arrive already expanded (`gfx::lines::build_vertices`):
//! world-space quads whose width was sized off each corner's depth, so a line
//! holds its pixel thickness at any distance. Like the decal pass this one
//! attaches no depth buffer and instead samples the scene depth, so an occluded
//! line fades to `OCCLUDED_ALPHA` rather than being clipped by hardware.
//!
//! Mirrors src/metal/line.rs.

use concinnity_core::gfx::render_types::LineVertex;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::fullscreen::align_up;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::DescriptorTables;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::error::map_hresult;
use crate::directx::pso::{Blend, GraphicsPso, Raster};
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::HDR_FORMAT;
use crate::directx::upload_ring::{UPLOAD_ALIGN, UploadRing};

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the lines readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// `LineView` is a GPU-free layout struct that lives in `core::render`;
// re-export it so `crate::directx::line::LineView` is the local path.
pub(in crate::directx) use concinnity_core::render::uniforms::LineView;

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
    depth_srv_gpu: SrvSlot,
}

impl LineResources {
    fn new(
        alloc: &DeviceAllocator,
        msaa_samples: u32,
        depth_srv_gpu: SrvSlot,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let (vs, ps) = compile_line_shaders(msaa_samples, hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_line_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_line_pso(device, &root_sig, &vs, &ps))?;

        let view_size = align256(std::mem::size_of::<LineView>() as u64);
        let frames = alloc.frames_in_flight();
        let mut view_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map line view ubo"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        Ok(Self {
            root_sig,
            pso,
            view_ubo_resources,
            view_ubo_ptrs,
            vertices: UploadRing::new(frames),
            depth_srv_gpu,
        })
    }
}

// Compile the line vertex + fragment shaders; the MSAA variant keeps the
// fragment shader's depth SRV declaration in sync with the resource's sample
// count. Used by the lazy build and by shader hot-reload.
fn compile_line_shaders(msaa_samples: u32, hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = builtin_shaders::LINE_VERT.compile(hot_reload)?;
    let ps = builtin_shaders::LINE_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
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
) -> RenderResult<ID3D12PipelineState> {
    let (vs, ps) = compile_line_shaders(msaa_samples, hot_reload)?;
    dump_on_err(
        info_queue,
        create_line_pso(device, &lines.root_sig, &vs, &ps),
    )
}

// Root-signature layout (binds 1:1 with the `line.hlsl` declarations, whose
// `register()` annotations `DXIL_ENTRY_ABI` in build.rs pins):
//   [0] root CBV b0   LineView (per-frame)
//   [1] table  t0     scene depth SRV (Texture2D[MS]<float>)
// No sampler: the fragment shader `Load`s the depth texel under the pixel.
fn create_line_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .srv_table(0, 1, Visibility::Pixel)
        .input_layout()
        .build(device, "line root sig")
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
) -> RenderResult<ID3D12PipelineState> {
    let layout = line_input_layout();
    GraphicsPso::new(root_sig, vs, ps)
        .target(HDR_FORMAT, Blend::AlphaOver)
        .input_layout(&layout)
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "line")
}

// Byte length of `vertex_count` ribbon vertices, as the `u32` a vertex buffer
// view carries. Errors rather than truncating a frame too large to bind.
fn line_vertex_bytes(vertex_count: usize) -> RenderResult<u32> {
    vertex_count
        .checked_mul(std::mem::size_of::<LineVertex>())
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| {
            RenderError::Other(format!(
                "line pass: {vertex_count} vertices exceed a vertex buffer view"
            ))
        })
}

// This frame's ribbon vertices as placed in the line upload ring: plain data
// the main thread hands the Lines pass, so the pass itself never touches the
// ring or the allocator behind it.
#[derive(Clone, Copy)]
pub(in crate::directx) struct LineUpload {
    gpu_va: u64,
    byte_len: u32,
}

impl LineUpload {
    fn new(gpu_va: u64, byte_len: u32) -> Self {
        Self { gpu_va, byte_len }
    }

    fn vertex_count(&self) -> u32 {
        self.byte_len / std::mem::size_of::<LineVertex>() as u32
    }

    fn vertex_buffer_view(&self) -> D3D12_VERTEX_BUFFER_VIEW {
        D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: self.gpu_va,
            SizeInBytes: self.byte_len,
            StrideInBytes: std::mem::size_of::<LineVertex>() as u32,
        }
    }
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
        let info_queue = self.hw.info_queue.clone();
        match LineResources::new(
            &self.hw.alloc,
            self.targets.hdr.msaa_samples,
            self.targets.main_depth_srv_gpu,
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

    // Copy this frame's expanded ribbons into this frame's slot of the upload
    // ring. Call once per frame on the main thread, before the graph fans its
    // passes out: growing the ring allocates through the device allocator,
    // which no worker may touch. The frame fence (waited before the slot is
    // reused) already retired the lists that read it last trip. `None` when
    // there is nothing to draw.
    pub(in crate::directx) fn upload_lines(
        &self,
        frame_idx: usize,
        vertices: &[LineVertex],
    ) -> RenderResult<Option<LineUpload>> {
        let Some(lines) = self.lines.resources.as_ref() else {
            return Ok(None);
        };
        if vertices.is_empty() {
            return Ok(None);
        }
        let byte_len = line_vertex_bytes(vertices.len())?;
        lines.vertices.reserve(
            &self.hw.alloc,
            frame_idx,
            align_up(u64::from(byte_len), UPLOAD_ALIGN),
        )?;
        let gpu_va = lines
            .vertices
            .push(frame_idx, bytemuck::cast_slice(vertices))?;
        Ok(Some(LineUpload::new(gpu_va, byte_len)))
    }

    // Encode the line pass: one unindexed triangle list covering every expanded
    // ribbon, alpha-blended into the resolved HDR target. `vp` is the same
    // view-projection the main pass rasterized with (jittered under TAA), so a
    // line sits on the pixel its geometry did. `upload` is this frame's ribbon
    // vertices, already in the ring (`upload_lines`).
    pub(in crate::directx) fn encode_lines(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        upload: Option<LineUpload>,
    ) {
        let (Some(lines), Some(upload)) = (self.lines.resources.as_ref(), upload) else {
            return;
        };

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
        let view_gva = com::gpu_va(&lines.view_ubo_resources[frame_idx]);

        // Main depth is already in a shader-resource state for the fragment's
        // occlusion sample: the graph declares this pass's depth read and the
        // executor emits the transition ahead of this command list.

        // The scene spine is a graph resource: this pass declares its
        // read-modify-write, so the executor has already put it in RENDER_TARGET
        // and the next consumer's barrier takes it back out.
        let scene_rtv = self.hdr_scene_rtv();

        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;
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
            cmd.IASetVertexBuffers(0, Some(&[upload.vertex_buffer_view()]));

            cmd.SetPipelineState(&lines.pso);
            cmd.SetGraphicsRootSignature(&lines.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.set_graphics_srv_table(1, lines.depth_srv_gpu);
            cmd.DrawInstanced(upload.vertex_count(), 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: u32 = std::mem::size_of::<LineVertex>() as u32;

    #[test]
    fn line_vertex_bytes_is_count_times_stride() {
        assert_eq!(line_vertex_bytes(0).unwrap(), 0);
        assert_eq!(line_vertex_bytes(6).unwrap(), 6 * STRIDE);
    }

    #[test]
    fn line_vertex_bytes_accepts_the_largest_view() {
        let max = (u32::MAX / STRIDE) as usize;
        assert_eq!(line_vertex_bytes(max).unwrap(), max as u32 * STRIDE);
    }

    #[test]
    fn line_vertex_bytes_refuses_a_view_overflow() {
        let past = (u32::MAX / STRIDE) as usize + 1;
        assert!(line_vertex_bytes(past).is_err());
        assert!(line_vertex_bytes(usize::MAX).is_err());
    }

    #[test]
    fn line_upload_binds_what_was_pushed() {
        let upload = LineUpload::new(0x1000, 12 * STRIDE);
        assert_eq!(upload.vertex_count(), 12);
        let view = upload.vertex_buffer_view();
        assert_eq!(view.BufferLocation, 0x1000);
        assert_eq!(view.SizeInBytes, 12 * STRIDE);
        assert_eq!(view.StrideInBytes, STRIDE);
    }
}

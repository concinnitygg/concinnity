// src/directx/draw/composite.rs
//
// Composite + text overlay: tonemap (and optionally LUT-grade) the HDR scene
// target onto the swapchain backbuffer, then layer the text vertices on top.
// The composite pass samples `scene_srv` (the post-TAA image when TAA is on,
// the HDR scene SRV otherwise) plus bloom mip 0; the text pass appends each
// label's vertex / index geometry into this frame slot's persistent upload
// buffer (see [`TextUploadRing`]) and binds sub-views into it, so no per-frame
// GPU buffers are allocated.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT;

use crate::gfx::render_types::{CompositeParams, TextDrawCall, TextVertex};

use crate::directx::context::DxContext;
use crate::directx::upload_ring::UPLOAD_ALIGN;
use concinnity_core::gfx::render_types::TextUniforms;

use crate::directx::graph_exec::{CompositeRenderTarget, CompositeResolution};
use crate::directx::pipeline::COMPOSITE_ROOT_CONSTANTS;
use crate::directx::texture::transition_barrier;

// Per-invocation binding context for the composite pass. The back-buffer is a
// cheap COM-refcount clone so `Args` carries no borrow (the trait's associated
// type can't name a lifetime). `pub` because it is the `Args` associated type of
// the (cross-crate) `render::fullscreen::CompositeEncoder` impl below, so it
// cannot be more private than that public trait's interface.
pub struct DxCompositeArgs {
    back_buffer: ID3D12Resource,
    back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    width: u32,
    height: u32,
    frame_idx: usize,
    // The `ViewMode` discriminant when the frame visualizes a G-buffer channel,
    // 0 otherwise (Lit / Unlit / Wireframe all composite the scene).
    channel_view: u32,
}

// The composite + text orchestration lives once in `gfx::fullscreen`; this impl
// drives each step in D3D12. The back buffer enters in `PRESENT`, is transitioned
// to `RENDER_TARGET` for the draws, and is returned to `PRESENT` on exit; the HDR
// target is expected to already be in `PIXEL_SHADER_RESOURCE` (the main pass
// leaves it that way).
impl crate::gfx::fullscreen::CompositeEncoder for DxContext {
    type Rec = ID3D12GraphicsCommandList;
    type Args = DxCompositeArgs;

    fn begin_composite(&self, cmd: &Self::Rec, args: &Self::Args) {
        let to_rt = transition_barrier(
            &args.back_buffer,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[to_rt]) };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&args.back_buffer_rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: args.width as f32,
                Height: args.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: args.width as i32,
                bottom: args.height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
        }
    }

    fn composite_draw(&self, cmd: &Self::Rec, args: &Self::Args) {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(&self.composite_pso);
            cmd.SetGraphicsRootSignature(&self.composite_root_sig);
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            // Root param [0]: scene SRV (t0): the TAA output when TAA is on,
            // the HDR scene target otherwise.
            cmd.SetGraphicsRootDescriptorTable(0, args.scene_srv);
            // Root param [1]: bloom mip 0 SRV (t1).
            cmd.SetGraphicsRootDescriptorTable(1, self.bloom.mip_srv_gpus[0]);
            // Root param [2]: CompositeParams (the post-process tunables plus
            // the scene-transition fade, matching the root-sig declaration).
            // Pushed verbatim so the HLSL cbuffer reads the same byte order as
            // the Rust struct.
            let composite = CompositeParams {
                post: self.post_process,
                fade: self.scene_fade,
                view_mode: args.channel_view,
                far: self.view_far,
            };
            cmd.SetGraphicsRoot32BitConstants(
                2,
                COMPOSITE_ROOT_CONSTANTS,
                &composite as *const CompositeParams as *const std::ffi::c_void,
                0,
            );
            // Root param [3]: 3D colour-grading LUT SRV (t2).
            cmd.SetGraphicsRootDescriptorTable(3, self.color_lut.srv_gpu);
            // Root params [4..6]: the G-buffer channel sources the debug view
            // modes visualize (t3 normal+depth, t4 roughness, t5 SSAO). The
            // fragment references all three statically, so they are bound on
            // every frame, channel view or not.
            let (nd_srv, rough_srv) = match self.gbuffer.as_ref() {
                Some(g) => (g.normal_depth_srv_gpu, g.roughness_srv_gpu),
                None => (self.ssao.white_srv_gpu, self.ssao.white_srv_gpu),
            };
            cmd.SetGraphicsRootDescriptorTable(4, nd_srv);
            cmd.SetGraphicsRootDescriptorTable(5, rough_srv);
            cmd.SetGraphicsRootDescriptorTable(6, self.ssao_ao_srv_gpu());
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            // The composite VS builds the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }

    fn begin_text(&self, cmd: &Self::Rec, args: &Self::Args) -> bool {
        let Some(text_pso) = &self.text_pso else {
            return false;
        };
        if self.descriptors.text_atlas_srv_gpus.is_empty() {
            return false;
        }
        // Root constants for the text pass (16 bytes = 4 DWORDs).
        let text_push = TextUniforms {
            win_width: args.width as f32,
            win_height: args.height as f32,
            _pad: [0.0; 2],
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(text_pso);
            cmd.SetGraphicsRootSignature(&self.text_root_sig);
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            cmd.SetGraphicsRoot32BitConstants(
                0,
                4,
                &text_push as *const TextUniforms as *const std::ffi::c_void,
                0,
            );
            cmd.SetGraphicsRootDescriptorTable(2, self.descriptors.text_sampler_gpu);
        }
        true
    }

    fn text_draw(
        &self,
        cmd: &Self::Rec,
        args: &Self::Args,
        call: &TextDrawCall,
    ) -> Result<(), String> {
        if call.vertices.is_empty() || self.descriptors.text_atlas_srv_gpus.is_empty() {
            return Ok(());
        }

        // Scissor a clipped (scrollable-panel) call to its band, restoring the
        // full-window scissor for an unclipped call so chrome is never cropped.
        // Windows client pixels are the overlay units, so this is a pure clamp.
        let ui = (args.width as f32, args.height as f32);
        let scissor = match call.clip_rect {
            Some(clip) => {
                match crate::gfx::fullscreen::clip_rect_to_scissor(
                    clip,
                    ui,
                    (args.width, args.height),
                ) {
                    // Row scrolled fully out of its band: nothing to draw.
                    None => return Ok(()),
                    Some((x, y, w, h)) => RECT {
                        left: x,
                        top: y,
                        right: x + w as i32,
                        bottom: y + h as i32,
                    },
                }
            }
            None => RECT {
                left: 0,
                top: 0,
                right: args.width as i32,
                bottom: args.height as i32,
            },
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };

        let atlas_idx = call
            .atlas_slot
            .min(self.descriptors.text_atlas_srv_gpus.len() - 1);

        // Append this label's vertex + index geometry into the frame slot's
        // persistent upload buffer (sized up front by `reserve` in
        // `encode_composite_and_text`) and bind sub-views into it.
        let vert_bytes = bytemuck::cast_slice(&call.vertices);
        let idx_bytes = bytemuck::cast_slice(&call.indices);

        let vert_va = self.text_upload.push(args.frame_idx, vert_bytes)?;
        let idx_va = self.text_upload.push(args.frame_idx, idx_bytes)?;

        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vert_va,
            SizeInBytes: vert_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<TextVertex>() as u32,
        };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: idx_va,
            SizeInBytes: idx_bytes.len() as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetGraphicsRootDescriptorTable(1, self.descriptors.text_atlas_srv_gpus[atlas_idx]);
            cmd.IASetVertexBuffers(0, Some(&[vbv]));
            cmd.IASetIndexBuffer(Some(&ibv));
            cmd.DrawIndexedInstanced(call.indices.len() as u32, 1, 0, 0, 0);
        }
        self.inc_draw_calls(1);
        Ok(())
    }

    fn end_composite(&self, cmd: &Self::Rec, args: &Self::Args) {
        let to_present = transition_barrier(
            &args.back_buffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[to_present]) };
    }
}

impl DxContext {
    // Encode the composite + text passes into `cmd` via the shared
    // `gfx::fullscreen` driver. Transitions the back buffer to `RENDER_TARGET`
    // for the draws and back to `PRESENT` on exit; the HDR target is expected to
    // already be in `PIXEL_SHADER_RESOURCE` (the main pass leaves it that way).
    pub(in crate::directx) fn encode_composite_and_text(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        render_target: CompositeRenderTarget<'_>,
        text_calls: &[TextDrawCall],
        scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
        resolution: CompositeResolution,
    ) -> Result<(), String> {
        let CompositeRenderTarget {
            back_buffer,
            back_buffer_rtv,
        } = render_target;
        let CompositeResolution { width, height } = resolution;
        // Reset this slot's text-upload cursor and ensure its buffer holds the
        // whole frame's text up front, so each `text_draw` only appends (and
        // never reallocates out from under an already-bound sub-view). The frame
        // fence in `draw_frame` has already confirmed the GPU is done with this
        // slot, so resetting / growing it now is race-free.
        let text_bytes = crate::gfx::fullscreen::text_upload_bytes(text_calls, UPLOAD_ALIGN);
        self.text_upload
            .reserve(&self.alloc, frame_idx, text_bytes)?;

        let args = DxCompositeArgs {
            back_buffer: back_buffer.clone(),
            back_buffer_rtv,
            scene_srv,
            width,
            height,
            frame_idx,
            channel_view: if self.view_mode.is_gbuffer_channel() {
                self.view_mode as u32
            } else {
                0
            },
        };
        crate::gfx::fullscreen::encode_composite_chain(self, cmd, &args, text_calls)
    }
}

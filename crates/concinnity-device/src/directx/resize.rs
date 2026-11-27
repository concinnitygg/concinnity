//! D3D12 swapchain + render-target resize handler. Polls the window's client size
//! at the top of each frame and, when the dimensions diverged from the live
//! render-target sizing, rebuilds every render-resolution-sized GPU resource and
//! rewrites the descriptors that point at them. The descriptor *slots* never move
//! (only the resources they point at) so the live root signatures + pipelines +
//! pre-bound GPU descriptor handles keep working without a re-bind.
//!
//! A `wait_idle` gate, a wholesale drop + recreate, then per-effect resource
//! rebuilds (TAA / SSAO / SSR / bloom). The shared post passes take their
//! targets' descriptors from the post block, so a pass whose target count
//! follows the resolution (the bloom chain's octaves) moves no other slot.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::render_graph::{PoolGates, plan_pool_slots};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::{BACK_BUFFERS, DxContext};
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::error::map_hresult;
use crate::directx::texture::{
    HDR_FORMAT, create_hdr_color_target, create_hdr_resolve_target, create_main_depth_texture,
    write_hdr_srv,
};
use concinnity_engine::render::post::device::PostExtent;

impl DxContext {
    // Rebuild every render-target-sized resource if the client area changed. A
    // minimized window skips the resize until a non-zero size returns.
    pub(super) fn maybe_handle_resize(&mut self) -> RenderResult<()> {
        let (width, height) = self.platform_window().framebuffer_size();
        let new_w = u32::saturating_from(width.max(0));
        let new_h = u32::saturating_from(height.max(0));
        if new_w == 0 || new_h == 0 {
            return Ok(());
        }
        // Compare against the *drawable* dims, not the render dims; with
        // temporal upscaling on, `extent.render_width`/`extent.render_height` are a
        // fraction of the window size and would never equal it, firing a
        // pointless rebuild every frame.
        if new_w == self.targets.extent.output_width && new_h == self.targets.extent.output_height {
            return Ok(());
        }
        self.handle_resize(new_w, new_h)
    }

    // (Re)acquire the swapchain back buffers and write their RTVs into the
    // pre-reserved RTV heap slots. Used by the resize path on both success (the
    // freshly-sized buffers) and failure (the unchanged old buffers), so a
    // failed `ResizeBuffers` never leaves `swapchain.back_buffers` empty.
    fn populate_back_buffers(&mut self) -> RenderResult<()> {
        self.swapchain.back_buffers.clear();
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let rtv_base = unsafe { self.swapchain.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in 0..BACK_BUFFERS {
            // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters
            // it fills are live locals that outlive the call.
            let buf: ID3D12Resource =
                unsafe { self.swapchain.handle.GetBuffer(u32::saturating_from(i)) }
                    .map_err(|e| map_hresult(e.code(), &format!("GetBuffer[{i}]")))?;
            let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_base.ptr + i * self.swapchain.rtv_descriptor_size,
            };
            // SAFETY: the view descriptor and the resource it names are live for the call, and the
            // destination handle addresses a slot this context reserved for the view in a heap it
            // owns.
            unsafe {
                self.hw
                    .device
                    .CreateRenderTargetView(&buf, None, rtv_handle);
            };
            self.swapchain.back_buffers.push(buf);
        }
        Ok(())
    }

    // Resize to a validated new size: wait idle, recreate at the new resolution,
    // and rewrite every dependent view in its existing heap slot.
    fn handle_resize(&mut self, new_w: u32, new_h: u32) -> RenderResult<()> {
        self.wait_idle();
        // Idle, so a post target this resize replaces returns its descriptor
        // slots at once.
        let _idle = self.post.idle_scope();

        // The upscaler bakes its sizes at creation, so recreate it for the new output
        // and derive the render dims from it; its views keep their slots. A failure
        // falls back to native resolution.
        let (render_w, render_h) = if let Some(old) = self.upscale.backend.as_ref() {
            let scale = old.extent().scale;
            let descriptors = old.output().descriptors();
            // Drop the old context before building the replacement (its
            // max_render / max_upscale sizes are baked at creation).
            self.upscale.backend = None;
            let rebuilt = crate::directx::post::upscale::build_upscaler(
                crate::directx::post::upscale::UpscaleDevice {
                    device: &self.hw.device,
                    streamline: self.hw.streamline.as_ref(),
                },
                (new_w, new_h),
                scale,
                descriptors,
                &mut self.upscale.requested,
            )?;
            let dims = match &rebuilt {
                Some(u) => u.extent().render,
                None => (new_w, new_h),
            };
            self.upscale.backend = rebuilt;
            dims
        } else {
            (new_w, new_h)
        };

        // `ResizeBuffers` needs every back-buffer reference gone, including those
        // recorded on each slot's end list, so reset and close them first (the GPU is
        // idle).
        for i in 0..self.hw.frames() {
            let allocator = &self.commands.end_command_allocators[i];
            let list = &self.commands.end_command_lists[i];
            // SAFETY: the fence for this frame slot was already waited on, so no submission still
            // references the allocator being reset.
            if let Err(e) = unsafe { allocator.Reset() } {
                tracing::warn!("resize: could not reset an end command allocator: {e}");
            }
            // SAFETY: the slot's GPU work is drained, so the list is not executing, and the
            // allocator it records into is live.
            if unsafe { list.Reset(allocator, None) }.is_err() {
                continue;
            }
            // SAFETY: the reset above left the list recording, which is what `Close` requires.
            if let Err(e) = unsafe { list.Close() } {
                tracing::warn!("resize: could not close an end command list: {e}");
            }
        }
        self.swapchain.back_buffers.clear();
        // ResizeBuffers must be passed the same flags the swapchain was created
        // with.
        let resize_flags = crate::directx::init::swapchain_flags(self.swapchain.allow_tearing);
        // SAFETY: the swapchain is live, every back-buffer reference this context held was dropped
        // above, and the call takes only scalars besides the format and flags.
        if let Err(e) = unsafe {
            self.swapchain.handle.ResizeBuffers(
                u32::saturating_from(BACK_BUFFERS),
                new_w,
                new_h,
                self.swapchain.format,
                resize_flags,
            )
        } {
            // On failure the swapchain keeps its size, so re-acquire the old back buffers
            // and retry next frame.
            self.populate_back_buffers()?;
            return Err(map_hresult(e.code(), "ResizeBuffers"));
        }
        self.populate_back_buffers()?;
        // The recreated back buffers invalidate any previously captured present
        // index; a `screenshot` before the next present then returns a clean
        // error instead of reading a stale buffer.
        self.swapchain.last_present_index = None;

        // The HDR color, resolve and depth; their SRVs are rewritten in place.
        self.targets.hdr.color = create_hdr_color_target(
            &self.hw.device,
            render_w,
            render_h,
            self.targets.hdr.msaa_samples,
            self.targets.hdr.color_rtv,
            self.state.view.clear_color,
        )?;
        if self.targets.hdr.msaa_samples > 1 {
            let resolve = create_hdr_resolve_target(&self.hw.device, render_w, render_h)?;
            // Rewrite the resolve target's RTV at the slot the decal pass
            // already binds. The CPU handle was captured at init when MSAA
            // turned the slot on; it stays valid across resize.
            if let Some(rtv) = self.targets.hdr.resolve_rtv {
                let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: HDR_FORMAT,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                };
                // SAFETY: the view descriptor and the resource it names are live for the call, and
                // the destination handle addresses a slot this context reserved for the view in a
                // heap it owns.
                unsafe {
                    self.hw
                        .device
                        .CreateRenderTargetView(&resolve, Some(&raw const rtv_desc), rtv);
                };
            }
            self.targets.hdr.resolve = Some(resolve);
        }
        self.targets
            .reactive_mask
            .resize(&self.hw.device, (render_w, render_h))?;
        // Recreate main depth (shader-readable so the decal/fog/auto-exposure
        // paths can sample it). The DSV is rewritten at the same slot.
        self.targets.depth.resource = create_main_depth_texture(
            &self.hw.device,
            render_w,
            render_h,
            self.targets.depth.dsv,
            self.targets.hdr.msaa_samples,
            true,
        )?;

        // Cache the SRV-heap CPU/GPU bases so the per-resource SRV rewrites
        // below can derive the CPU handle from each stored GPU handle without
        // borrowing `self` again (the per-effect rebuilds need `&mut self`).
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let srv_cpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        let srv_gpu_base = SrvSlot::at(
            &self.descriptors.srv_heap,
            self.descriptors.srv_descriptor_size,
            0,
        );
        let srv_cpu_of = |gpu: SrvSlot| gpu.cpu_in(srv_cpu_base, srv_gpu_base);

        // 3) Refresh SRVs that point at the recreated resources. The GPU
        //    handles stored on the various Resources structs already match
        //    these heap slots; we just rewrite the descriptors in place.
        write_hdr_srv(
            &self.hw.device,
            self.targets
                .hdr
                .resolve
                .as_ref()
                .unwrap_or(&self.targets.hdr.color),
            srv_cpu_of(self.targets.hdr.srv_gpu),
        );

        // The main-depth SRV is shared by the decal pass and the fog pass.
        // Both store the same `depth_srv_gpu`; rewrite the descriptor once
        // and both consumers pick it up.
        if let Some(decals) = self.decal.state.as_ref() {
            crate::directx::decal::write_main_depth_srv(
                &self.hw.device,
                &self.targets.depth.resource,
                srv_cpu_of(decals.depth_srv_gpu),
                self.targets.hdr.msaa_samples,
            );
        }

        // Rebuild the transient pool first; bloom and SSAO re-view their targets from
        // it.
        self.targets.transient_pool.rebuild(
            &self.hw.device,
            &self.hw.command_queue,
            &plan_pool_slots(PoolGates::REACHABLE, (render_w, render_h), (new_w, new_h))?,
        )?;

        // 4) Bloom: the octaves below the pool's top one, and the view of the
        //    rebuilt `bloom_top`.
        let bloom_top = self
            .targets
            .transient_pool
            .resource_for("bloom_top")
            .ok_or_else(|| RenderError::Other("transient pool missing bloom_top on resize".into()))?
            .clone();
        let output = PostExtent {
            width: new_w,
            height: new_h,
        };
        if let Some(mut bloom) = self.bloom.take() {
            let r = bloom.resize(&self.post_device(0), output, &bloom_top);
            self.bloom = Some(bloom);
            r?;
        }

        // TAA restarts its history: reprojection from the old resolution is invalid.
        if let Some(mut taa) = self.taa.take() {
            let r = taa.resize_to(&self.post_device(0), render_w, render_h);
            self.taa = Some(taa);
            r?;
        }

        // 6) SSAO: the raw occlusion. The blurred `ao_output` is pooled and was
        // rebuilt above; SSAO re-views it.
        if let Some(ao_resource) = self
            .targets
            .transient_pool
            .resource_for("ao_output")
            .cloned()
            && let Some(mut ssao) = self.ssao.resources.take()
        {
            let r = ssao.resize(
                &self.post_device(0),
                PostExtent {
                    width: render_w,
                    height: render_h,
                },
                &ao_resource,
            );
            self.ssao.resources = Some(ssao);
            r?;
        }

        // 7) SSR: the reflection target the resolve writes.
        if let Some(mut ssr) = self.ssr.take() {
            let r = ssr.resize_to(&self.post_device(0), render_w, render_h);
            self.ssr = Some(ssr);
            r?;
        }

        // 7-gbuffer) Unified G-buffer pre-pass: the three color targets are
        // pooled and were relocated by the rebuild above, so this re-points
        // their views; the private depth is feature-owned and recreated.
        if let Some(pooled) = self.targets.transient_pool.gbuffer_pooled()
            && let Some(gbuffer) = self.gbuffer.as_mut()
        {
            gbuffer.resize_to(
                &self.hw.device,
                render_w,
                render_h,
                srv_cpu_base,
                srv_gpu_base,
                &pooled,
            )?;
        }

        // 7-ssgi) SSGI pyramids and accumulation.
        if let Some(mut ssgi) = self.ssgi.take() {
            let r = ssgi.resize_to(&self.post_device(0), render_w, render_h);
            self.ssgi = Some(ssgi);
            r?;
        }
        self.sync_disocclusion()?;

        // 7-rt) RT reflections output target, into its own descriptors; the
        //     acceleration structure is resolution-independent and is left
        //     untouched.
        if let Some(rt) = self.rt_reflections.as_mut() {
            rt.resize_to(&self.hw.device, &self.post, render_w, render_h)?;
        }

        // 7-refl) Reflection composite: the full-res composited output + the
        //     reduced-res roughness blur. Every consumer reads the output's SRV
        //     per frame, so the new targets need no re-bind.
        if let Some(mut rc) = self.reflection_composite.take() {
            let r = rc.resize(
                &self.post_device(0),
                PostExtent {
                    width: render_w,
                    height: render_h,
                },
            );
            self.reflection_composite = Some(rc);
            r?;
        }

        // 7a) Raymarch: recreate the `hdr_resolve_copy` scene snapshot at
        //     the new dims and rewrite its SRV descriptor in place. The
        //     descriptor slot itself doesn't move, so the live raymarch
        //     root-table binding stays valid without a re-bind.
        if let Some(rm) = self.raymarch.as_mut() {
            rm.resize_to(&self.hw.device, render_w, render_h)?;
        }

        // The Hi-Z pyramid, in its reserved slots, is invalid until rebuilt, so the
        // next cull skips occlusion.
        if let Some(hiz) = self.cull.hiz.as_mut() {
            hiz.resize_to(&self.hw.device, render_w, render_h)?;
        }
        if let Some(receivers) = self.shadow_receivers.as_mut() {
            receivers.resize(&self.hw.device, [render_w, render_h])?;
        }

        // 7b') The clouds' target follows the render size; its SRV and UAV are
        //      rewritten in place, so their slots stay valid.
        if let Some(clouds) = self.atmosphere.as_mut().and_then(|a| a.clouds.as_mut()) {
            clouds.resize_to(&self.hw.device, [render_w, render_h])?;
        }
        self.state.planner.invalidate_occlusion();

        // 7c) The transparent pass: recreate the scene snapshot at the new dims
        //     and rewrite its SRV in place. The depth SRV the pass also binds is
        //     the main-depth slot, rewritten by the decal path.
        if let Some(transparent) = self.transparent.as_mut() {
            transparent.resize_to(&self.hw.device, &self.post, render_w, render_h)?;
        }

        // 7d) Planar reflections: recreate the shared mirror color + depth + the
        //     per-plane resolves at the new render dims and rewrite their RTV / DSV /
        //     SRVs in place, so each reflector's resolve binding stays valid.
        if let Some(planar) = self.planar_reflection.as_mut() {
            planar.resize_to(&self.hw.device, render_w, render_h)?;
        }

        // 8) Commit the new dimensions. `render_*` drives the scene-pass
        //    viewports + the sub-pixel jitter; `output_*` drives the
        //    composite viewport and is what the next resize poll compares
        //    against. They differ only while temporal upscaling is active.
        self.targets.extent.render_width = render_w;
        self.targets.extent.render_height = render_h;
        self.targets.extent.output_width = new_w;
        self.targets.extent.output_height = new_h;

        Ok(())
    }
}

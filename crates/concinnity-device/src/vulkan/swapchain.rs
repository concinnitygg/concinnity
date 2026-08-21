// src/vulkan/swapchain.rs
//
// Vulkan swapchain, attachment, and framebuffer creation, plus the
// swapchain rebuild path.
use ash::vk;

use crate::vulkan::owned::{OwnedFramebuffer, VkDevice};

use super::allocator::DeviceAllocator;
use super::context::*;
use super::device::*;
use super::glass::{GlassDeviceCtx, GlassRebuildTargets};
use super::hiz::{HiZDeviceCtx, HiZTarget};
use super::post::bloom::{
    BloomDeviceContext, alloc_bloom_input_sets, create_bloom_chain, create_bloom_framebuffers,
    rebind_bloom_input0,
};
use super::post::gbuffer::{GbufferDeviceCtx, GbufferExtent, GbufferQueueCtx};
use super::post::reflection_composite::CompositeInputViews;
use super::post::rt_reflections::RtStaticInputs;
use super::post::ssao::SsaoDeviceCtx;
use super::post::ssgi::SsgiDevice;
use super::post::ssr::{SsrExtent, SsrGpuContext, SsrResolveInputs};
use super::post::taa::{TaaDeviceContext, TaaSceneInputs};
use super::post::upscale::UpscalerGpu;
use super::raymarch::RaymarchDeviceContext;
use super::texture::*;

//  Swapchain rebuild

impl VkContext {
    pub(super) fn destroy_swapchain_resources(&mut self) {
        let device = &self.device;
        for iv in &self.swapchain.image_views {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe { device.destroy_image_view(*iv, None) };
        }
        // On a `reload_world` the successor context inherits this swapchain
        // (Vulkan handles are not refcounted), so the outgoing context frees only
        // its own image views / attachments above and leaves the swapchain
        // itself alive. Always false during a normal resize rebuild (the only
        // other caller), so a resize still recreates the swapchain as before.
        if !self.reused_by_successor {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe {
                self.swapchain
                    .loader
                    .destroy_swapchain(self.swapchain.handle, None)
            };
        }
        self.framebuffers.clear();
        self.composite.framebuffers.clear();
        self.bloom.write_framebuffers.clear();
        self.bloom.blend_framebuffers.clear();
        // Dropping the attachment images (and the bloom mips; a borrowed
        // pooled mip 0 releases nothing) retires them through the allocator.
        self.bloom.mips.clear();
        self.bloom.mip_extents.clear();
        self.color_images.clear();
        self.depth_images.clear();
        self.hdr_resolve_images.clear();
        self.swapchain.image_views.clear();
    }

    // The extent a rebuild would create the swapchain at, read from the surface
    // instead of the window. See `rebuild_swapchain` for why the distinction
    // matters.
    pub(super) fn surface_extent(&self) -> Result<vk::Extent2D, String> {
        // SAFETY: a property query on a live handle; it only reads.
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(|e| format!("surface caps: {e}"))?;
        let (width, height) = self.window().framebuffer_size();
        Ok(resolve_swapchain_extent(
            &caps,
            width.max(0) as u32,
            height.max(0) as u32,
        ))
    }

    pub(super) fn rebuild_swapchain(&mut self) -> Result<(), String> {
        // A minimised window has a 0x0 client area, and a zero-extent swapchain
        // (with every attachment / framebuffer sized from it) is invalid. Skip
        // the rebuild and leave the existing resources at their last non-zero
        // size; a later frame rebuilds once the window is restored. Mirrors
        // DirectX `maybe_handle_resize`'s minimise skip.
        if self.is_minimized() {
            return Ok(());
        }
        // The window's cached client size is not enough on its own: it holds
        // whatever the last WM_SIZE delivered, and a present returning
        // SUBOPTIMAL / OUT_OF_DATE rebuilds inside the same frame, before the
        // pump that would report the minimise. The surface reports 0x0 straight
        // away and is where the extent actually comes from, so gate on it too.
        // Vsync turns that race into the common case, since the frame blocks in
        // FIFO present for as long as the minimise takes to arrive.
        if !extent_is_presentable(self.surface_extent()?) {
            return Ok(());
        }
        self.wait_idle();
        // The previous swapchain's images are about to be destroyed; invalidate
        // the screenshot read-back index until the next present repopulates it.
        self.swapchain.last_present_index = None;
        self.destroy_swapchain_resources();

        let (width, height) = self.window().framebuffer_size();
        // re-query present family
        let present_family = {
            let (_, pf) = query_queue_families(
                &self.instance,
                self.physical_device,
                &self.surface_loader,
                self.surface,
            )?;
            pf
        };
        let (sc, imgs, fmt, ext) = create_swapchain_inner(
            &SwapchainSurface {
                instance: &self.instance,
                device: &self.device,
                pd: self.physical_device,
                surface_loader: &self.surface_loader,
                surface: self.surface,
                swapchain_loader: &self.swapchain.loader,
            },
            SwapchainQueueFamilies {
                graphics_family: self.graphics_family,
                present_family,
            },
            SwapchainConfig {
                width: width as u32,
                height: height as u32,
                old_swapchain: vk::SwapchainKHR::null(),
                hdr_mode: self.hdr_mode,
                vsync: self.vsync,
            },
        )?;
        self.swapchain.handle = sc;
        self.swapchain.images = imgs;
        self.swapchain.format = fmt;
        self.swapchain.extent = ext;
        // Temporal upscaling: the FSR context bakes its max render / upscale
        // sizes at creation, so a resize must recreate it at the new output
        // size (same quality scale). `device_wait_idle` at the top of this
        // function guarantees the old context is idle before destroy. The new
        // render dims then drive `render_ext`; off-screen scene passes rebuild
        // to it while bloom / composite / swapchain stay at `ext`.
        if let Some(scale) = self.upscale.as_ref().map(|u| u.scale()) {
            if let Some(mut old) = self.upscale.take() {
                old.destroy(&self.device);
            }
            // Rebuild the backend the world requested (not a hardcoded FSR). The
            // DLSS / XeSS device extensions are fixed at device creation, and
            // `build_upscaler` re-resolves `upscale_requested` deterministically
            // to the same first choice, so the rebuilt backend matches the device.
            let (built, resolved) = super::post::build_upscaler(
                UpscalerGpu {
                    alloc: &self.alloc,
                    instance: &self.instance,
                    device: &self.device,
                    physical_device: self.physical_device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                ext.width,
                ext.height,
                scale,
                self.upscale_requested,
            )?;
            // The rebuilt feature re-emits the benign DLSS first-frame layout
            // errors; re-arm the messenger budget so they stay suppressed.
            if resolved == super::post::ResolvedBackend::Dlss
                && let Some(f) = self.device.debug_filter()
            {
                f.store(
                    super::init::DLSS_FIRST_FRAME_LAYOUT_SUPPRESS,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            self.upscale = built;
        }
        let render_ext = match &self.upscale {
            Some(u) => {
                let (w, h) = u.render_dims();
                vk::Extent2D {
                    width: w,
                    height: h,
                }
            }
            None => ext,
        };
        self.render_extent = render_ext;

        // Rebuild the transient image pool before the off-screen attachments /
        // bloom chain / SSAO targets that bind its images. `ao_output` is
        // render-res, `bloom_top` is half the output (swapchain) extent; both are
        // per frame in flight. `bloom_top_pairs` feeds the bloom chain's mip 0
        // below (empty when bloom is off, so mip 0 is committed instead).
        self.transient_pool.rebuild(
            &super::transient_pool::TransientPoolGpu {
                instance: &self.instance,
                device: &self.device,
                physical_device: self.physical_device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            self.frames_in_flight,
            &super::transient_pool::transient_slots(
                self.ssao.is_some(),
                self.post_process.bloom_intensity > 0.0,
                self.gbuffer.is_some(),
                render_ext,
                ext,
            )?,
        )?;
        let bloom_top_pairs = self
            .transient_pool
            .pairs_for_frames("bloom_top", self.frames_in_flight);

        self.swapchain.image_views =
            create_swapchain_image_views(&self.device, &self.swapchain.images, fmt)?;

        let (color_images, depth_images, hdr_resolve_images) = create_attachments(
            &AttachmentDeviceCtx {
                alloc: &self.alloc,
                device: &self.device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            render_ext.width,
            render_ext.height,
            self.msaa_samples,
            self.frames_in_flight,
        )?;
        self.color_images = color_images;
        self.depth_images = depth_images;
        self.hdr_resolve_images = hdr_resolve_images;
        self.framebuffers = create_main_framebuffers(
            &self.device,
            self.main_render_pass.handle(),
            &self.color_images,
            &self.depth_images,
            &self.hdr_resolve_images,
            render_ext,
            self.msaa_samples,
        )?;
        self.composite.framebuffers = create_composite_framebuffers(
            &self.device,
            self.composite.render_pass.handle(),
            &self.swapchain.image_views,
            ext,
        )?;

        // Rebuild the bloom chain at the new resolution.
        let (bloom_mips, bloom_mip_extents) = create_bloom_chain(
            &BloomDeviceContext {
                alloc: &self.alloc,
                device: &self.device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            ext,
            self.frames_in_flight,
            &bloom_top_pairs,
        )?;
        self.bloom.mips = bloom_mips;
        self.bloom.mip_extents = bloom_mip_extents;
        let (bloom_write_framebuffers, bloom_blend_framebuffers) = create_bloom_framebuffers(
            &self.device,
            self.bloom.write_pass.handle(),
            self.bloom.blend_pass.handle(),
            &self.bloom.mips,
            &self.bloom.mip_extents,
        )?;
        self.bloom.write_framebuffers = bloom_write_framebuffers;
        self.bloom.blend_framebuffers = bloom_blend_framebuffers;

        // The bloom input sets reference the destroyed mips; reset the pool
        // (the octave count may have changed) and re-allocate. wait_idle()
        // above guarantees none are still in flight.
        // SAFETY: `descriptor_pool` was created from this device and every set allocated from it is
        // dropped here; the caller has already idled the device, so none is still in use.
        unsafe {
            self.device
                .reset_descriptor_pool(
                    self.bloom.descriptor_pool.handle(),
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(|e| format!("reset bloom pool: {e}"))?;
        }
        self.bloom.input_sets = alloc_bloom_input_sets(
            &self.device,
            self.bloom.descriptor_pool.handle(),
            self.bloom.set_layout.handle(),
            self.composite.sampler.handle(),
            &self.hdr_resolve_images,
            &self.bloom.mips,
        )?;

        // Rebuild the unified G-buffer pre-pass targets at the new resolution
        // *first*: every reader (SSR resolve, SSAO, SSGI, RT, TAA velocity, FSR)
        // re-points its descriptors at the rebuilt per-frame normal+depth /
        // roughness / velocity views below, so the merged buffer must already be
        // current. The render pass, pipelines, UBOs, and descriptor sets survive.
        if let Some(mut gb) = self.gbuffer.take() {
            // The three colour channels are pool-owned and were reallocated by
            // the pool rebuild above, so the framebuffers built here reference
            // the new images.
            let pooled = self.transient_pool.gbuffer_pooled(self.frames_in_flight);
            gb.rebuild(
                GbufferDeviceCtx {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                GbufferQueueCtx {
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                GbufferExtent {
                    width: render_ext.width,
                    height: render_ext.height,
                    frames: self.frames_in_flight,
                },
                &pooled,
            )?;
            self.gbuffer = Some(gb);
        }

        // Rebuild the SSR targets at the new resolution. The G-buffer +
        // roughness + private depth + output are all resolution-dependent;
        // the resolve sets re-point automatically at the new HDR resolve +
        // SSR targets via wire_resolve_sets. With SSR on, the bloom prefilter
        // input 0 also moves to the new SSR output below; TAA (when on)
        // overrides that in turn to the new TAA output.
        if let Some(mut ssr) = self.ssr.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // Per-frame unified G-buffer views (rebuilt above) when present, else
            // empty so the SSR resolve falls back to its own pre-pass targets.
            let (nd_views, rough_views) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views(), gb.roughness_views()),
                None => (Vec::new(), Vec::new()),
            };
            ssr.rebuild(
                &SsrGpuContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                SsrExtent {
                    width: render_ext.width,
                    height: render_ext.height,
                },
                SsrResolveInputs {
                    hdr_resolve_views: &hdr_views,
                    gbuffer_views: &nd_views,
                    roughness_views: &rough_views,
                    prefilter_view: self.env_map.prefilter.view,
                    cube_sampler: self.cube_sampler.handle(),
                },
            )?;
            // The bloom prefilter samples the reflection composite output (re-pointed
            // in the composite rebuild below), not the raw resolve output.
            self.ssr = Some(ssr);
        }

        // Rebuild the SSGI gi target + composite framebuffers and re-wire its
        // descriptor sets to the rebuilt HDR resolves + SSR pre-pass G-buffer.
        // The SSR rebuild above already ran, so `ssr.gbuffer` is current. The
        // render passes, pipelines, sampler, and descriptor pool all survive.
        if let Some(mut ssgi) = self.ssgi.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // SSGI samples the unified G-buffer's per-frame normal+depth views.
            // The merged pre-pass was rebuilt above, so they are current.
            let nd_views = self
                .gbuffer
                .as_ref()
                .expect("SSGI keeps the unified G-buffer pre-pass alive")
                .normal_depth_views();
            ssgi.rebuild(
                SsgiDevice {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                render_ext.width,
                render_ext.height,
                &hdr_views,
                &nd_views,
            )?;
            self.ssgi = Some(ssgi);
        }

        // Rebuild the RT-reflection output target + re-wire its static
        // descriptors (the SSR pre-pass G-buffer / roughness + the HDR resolves
        // all moved). The acceleration structure is resolution-independent, so it
        // survives; the per-frame TLAS + geometry-table descriptors are re-pointed
        // by `rt_dynamic_update` as usual. RT output is a single shared image, so
        // the bloom prefilter input 0 moves to it (TAA / upscale override below).
        if let Some(mut rt) = self.rt_reflections.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // RT samples the unified G-buffer's per-frame normal+depth + roughness
            // views. The merged pre-pass was rebuilt above, so they are current.
            let gb = self
                .gbuffer
                .as_ref()
                .expect("RT keeps the unified G-buffer pre-pass alive");
            let nd_views = gb.normal_depth_views();
            let rough_views = gb.roughness_views();
            rt.rebuild(
                &self.alloc,
                &self.device,
                render_ext.width,
                render_ext.height,
                RtStaticInputs {
                    vertex_buffer: self.geometry.vertex_buffer.buffer(),
                    index_buffer: self.geometry.index_buffer.buffer(),
                    hdr_resolve_views: &hdr_views,
                    gbuffer_views: &nd_views,
                    roughness_views: &rough_views,
                    prefilter_view: self.env_map.prefilter.view,
                    cube_sampler: self.cube_sampler.handle(),
                },
            )?;
            // The bloom prefilter samples the reflection composite output (re-pointed
            // in the composite rebuild below), not the raw RT output.
            self.rt_reflections = Some(rt);
        }

        // Rebuild the reflection composite's output + blur targets at the new
        // resolution + re-wire its static bindings (the rebuilt HDR resolves +
        // G-buffer views moved), then re-point the bloom prefilter input 0 at its
        // output (the scene image; TAA / upscale override below). The reflection
        // binding is re-pointed per encode, so the resolve rebuilds need no extra
        // wiring here.
        if let Some(mut rc) = self.reflection_composite.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let (nd_views, rough_views) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views(), gb.roughness_views()),
                None => (Vec::new(), Vec::new()),
            };
            rc.rebuild(
                &super::texture::GpuUploadContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                render_ext.width,
                render_ext.height,
                &CompositeInputViews {
                    hdr_resolve_views: &hdr_views,
                    normal_depth_views: &nd_views,
                    roughness_views: &rough_views,
                },
            )?;
            for frame_sets in &self.bloom.input_sets {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    rc.output.view,
                    self.composite.sampler.handle(),
                );
            }
            self.reflection_composite = Some(rc);
        }

        // Rebuild the TAA velocity + history targets at the new resolution.
        // When TAA is on the bloom prefilter + composite sample its output
        // image; otherwise they sample the raw HDR resolve (or SSR output
        // when SSR is on but TAA is off). wait_idle() above guarantees none
        // of these are still in flight.
        if let Some(mut taa) = self.taa.take() {
            taa.rebuild(
                &TaaDeviceContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                render_ext,
                self.frames_in_flight,
                &TaaSceneInputs {
                    hdr_resolve_images: &self.hdr_resolve_images,
                    sampler: self.composite.sampler.handle(),
                },
            )?;
            // When a reflection path owns the scene image, TAA samples the reflection
            // composite output (HDR + reflections) instead of the raw HDR resolve. A
            // SSGI-only build leaves TAA on the raw HDR resolve.
            if let Some(rc) = self.reflection_composite.as_ref() {
                taa.rewire_scene(
                    &self.device,
                    rc.output.view,
                    self.composite.sampler.handle(),
                );
            }
            // The TAA resolve's velocity input is the unified G-buffer's per-frame
            // velocity channel (rebuilt above), replacing TAA's own velocity
            // pre-pass output. Mirrors the init-time `rewire_velocity`.
            if let Some(gb) = self.gbuffer.as_ref() {
                let vel_views = gb.velocity_views();
                taa.rewire_velocity(&self.device, &vel_views, self.composite.sampler.handle());
            }
            for (i, frame_sets) in self.bloom.input_sets.iter().enumerate() {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    taa.output_view(i),
                    self.composite.sampler.handle(),
                );
            }
            self.taa = Some(taa);
        }

        // Temporal upscaling: bloom prefilter samples the FSR output (the
        // reconstructed swapchain-res scene), overriding the SSR / TAA rebinds
        // above. A single shared image, so every frame's set points at it.
        if let Some(up) = &self.upscale {
            let up_output_view = up.output_image().view;
            for frame_sets in &self.bloom.input_sets {
                rebind_bloom_input0(
                    &self.device,
                    frame_sets[0],
                    up_output_view,
                    self.composite.sampler.handle(),
                );
            }
        }

        // Rebuild the decal framebuffers at the new resolution + re-point
        // the per-frame depth descriptor at the rebuilt depth view. The
        // pipeline, layouts, buffers, sampler, and per-decal albedo sets
        // all survive: only the targets the framebuffers + depth binding
        // reference moved.
        if let Some(mut decals) = self.decal.resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            decals.rebuild(&self.device, &hdr_views, &depth_views, render_ext)?;
            self.decal.resources = Some(decals);
        }

        // Rebuild the line framebuffers + re-point the per-frame depth
        // descriptor. Mirrors the decal rebuild; only present once a frame
        // published lines and the lazy build ran.
        if let Some(mut lines) = self.lines.resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            lines.rebuild(&self.device, &hdr_views, &depth_views, render_ext)?;
            self.lines.resources = Some(lines);
        }

        // Rebuild the fog framebuffers + re-point the per-frame depth
        // descriptor at the rebuilt depth view. Mirrors the decal rebuild;
        // the pipeline, layouts, UBOs, and sampler all survive.
        if let Some(mut fog) = self.fog.resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            fog.rebuild(&self.device, &hdr_views, &depth_views, render_ext)?;
            self.fog.resources = Some(fog);
        }

        // Recreate the raymarch scene snapshot at the new resolution + re-point
        // the `scene_color` binding of every view set. The pipelines, layouts,
        // UBOs, cube buffers, and render passes survive; the pass reuses the
        // rebuilt main framebuffers, so only the snapshot moved.
        if let Some(mut rm) = self.raymarch.take() {
            rm.rebuild(
                RaymarchDeviceContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                render_ext.width,
                render_ext.height,
            )?;
            self.raymarch = Some(rm);
        }

        // Rebuild the glass scene snapshot + per-frame framebuffers at the new
        // resolution + re-point the snapshot / depth bindings. The scene target
        // moved with the rebuilt reflection composite output / HDR resolve, so
        // resolve it again here (composite output when a reflection path is active,
        // else this slot's HDR resolve). The composite + HDR resolve rebuilds above
        // already ran, so the handles are current. The pipeline, layouts, panel
        // buffers, view UBOs, and render pass all survive.
        // Planar reflection mirror targets follow the render resolution; rebuild
        // them before glass so the per-pane planar binding re-point below picks up
        // the new target views.
        if let Some(mut planar) = self.planar_reflection.take() {
            planar.rebuild(
                &self.alloc,
                &self.device,
                render_ext.width,
                render_ext.height,
            )?;
            self.planar_reflection = Some(planar);
        }
        let planar_target_views: Vec<vk::ImageView> = self
            .planar_reflection
            .as_ref()
            .map(|s| (0..s.plane_count()).map(|i| s.target_view(i)).collect())
            .unwrap_or_default();
        if let Some(mut glass) = self.glass.take() {
            let (scene_views, scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0..self
                .frames_in_flight)
                .map(|i| match self.reflection_composite.as_ref() {
                    Some(rc) => (rc.output.view, rc.output.image),
                    None => (
                        self.hdr_resolve_images[i].view,
                        self.hdr_resolve_images[i].image,
                    ),
                })
                .unzip();
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            glass.rebuild(
                GlassDeviceCtx {
                    alloc: &self.alloc,
                    instance: &self.instance,
                    device: &self.device,
                    physical_device: self.physical_device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                render_ext.width,
                render_ext.height,
                GlassRebuildTargets {
                    scene_views: &scene_views,
                    scene_images: &scene_images,
                    depth_views: &depth_views,
                    planar_target_views: &planar_target_views,
                },
            )?;
            self.glass = Some(glass);
        }

        // Rebuild the Hi-Z pyramid at the new resolution + re-point its init
        // sets' depth bindings and the cull-read set's pyramid sampler. The
        // build pipelines, layouts, sampler, and per-frame cull UBOs survive.
        // Invalidate the pyramid so the next frame's cull falls back to frustum
        // + distance until a pyramid at the new resolution has been built.
        if let Some(mut hiz) = self.cull.hiz.take() {
            let depth_views: Vec<vk::ImageView> =
                self.depth_images.iter().map(|img| img.view).collect();
            hiz.resize_to(
                HiZDeviceCtx {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                HiZTarget {
                    width: render_ext.width,
                    height: render_ext.height,
                    depth_views: &depth_views,
                },
            )?;
            self.cull.hiz = Some(hiz);
            self.cull.hiz_valid = false;
        }

        // Re-point the planar reflected-frustum cull's Hi-Z set at the freshly
        // rebuilt pyramid view. The Hi-Z resize above destroyed the view the planar
        // set captured at init; the persistent planar set must follow or its set 1
        // dangles a freed image view (bound every frame even though hiz_enabled = 0
        // keeps it unsampled). A no-op when there's no planar set or no Hi-Z.
        if let (Some(planar), Some(hiz)) = (self.planar_reflection.as_ref(), self.cull.hiz.as_ref())
        {
            let (view, sampler) = hiz.read_set_sources();
            planar.rewrite_hiz_view(&self.device, view, sampler);
        }

        // An in-flight probe bake's Hi-Z set captured the same destroyed view
        // at bake start; re-point it too or its next face binds a freed view.
        if let (Some(bake), Some(hiz)) = (self.probe.rendering.as_ref(), self.cull.hiz.as_ref()) {
            let (view, sampler) = hiz.read_set_sources();
            bake.rewrite_hiz_view(&self.device, view, sampler);
        }

        // Rebuild the particle framebuffers at the new resolution. The
        // pipelines, layouts, view UBOs, per-emitter pools, and
        // descriptor sets all survive: only the framebuffers reference
        // the moved hdr_resolve targets.
        if let Some(mut p) = self.particle.resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            p.rebuild(&self.device, &hdr_views, render_ext)?;
            self.particle.resources = Some(p);
        }

        // Re-point the auto-exposure build sets at the rebuilt HDR resolve
        // views. The histogram / output / readback buffers are
        // resolution-independent and survive the rebuild untouched.
        if let Some(mut ae) = self.auto_exposure.resources.take() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            ae.rebuild(&self.device, &hdr_views, self.linear_sampler.handle());
            self.auto_exposure.resources = Some(ae);
        }

        // Rebuild the SSAO targets + re-point the SSAO descriptor at set 0
        // binding 6 of every global set against the per-frame pooled `ao_output`
        // views (the transient pool was already rebuilt above). SSAO's stale
        // blur framebuffers are torn down inside `ssao.rebuild` (the device is
        // idle, so freeing the pool views ahead of those framebuffers is sound).
        // When SSAO is off the pool holds no `ao_output` and binding 6 stays on
        // the (resolution-independent) 1×1 white fallback, so no rebuild needed.
        let frames = self.frames_in_flight;
        if let Some(mut ssao) = self.ssao.take() {
            // SSAO kernel/blur sample the unified G-buffer's per-frame normal+depth
            // views (rebuilt above) when present, else SSAO's own pre-pass target.
            let nd_views = match self.gbuffer.as_ref() {
                Some(gb) => gb.normal_depth_views(),
                None => Vec::new(),
            };
            let ao_views = self.transient_pool.views_for_frames("ao_output", frames);
            ssao.rebuild(
                &SsaoDeviceCtx {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                render_ext.width,
                render_ext.height,
                &nd_views,
                &ao_views,
            )?;
            for (i, &set) in self.descriptors.global_sets.iter().enumerate() {
                let ao_view = self
                    .transient_pool
                    .view_for("ao_output", i)
                    .unwrap_or(self.ssao_white.view);
                let info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(ao_view)
                    .sampler(self.linear_sampler.handle());
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&info));
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
                // every set and resource it names belongs to this device.
                unsafe {
                    self.device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
            self.ssao = Some(ssao);
        }

        // Re-point the composite descriptor sets at the rebuilt scene-input
        // image (FSR upscale output > TAA output > reflection composite output >
        // HDR resolve) + bloom mip 0. The 3D colour LUT is resolution-independent,
        // so it survives the resize untouched and is just re-bound at binding 2.
        for (i, &set) in self.composite.sets.iter().enumerate() {
            let scene_view = if let Some(up) = &self.upscale {
                up.output_image().view
            } else if let Some(taa) = &self.taa {
                taa.output_view(i)
            } else if let Some(rc) = self.reflection_composite.as_ref() {
                rc.output.view
            } else {
                self.hdr_resolve_images[i].view
            };
            write_composite_set(
                &self.device,
                set,
                scene_view,
                self.bloom.mips[i][0].view,
                self.color_lut.view,
                self.composite.sampler.handle(),
            );
            // The view-mode channel sources are resolution-dependent too, so
            // they follow the rebuilt G-buffer / AO targets.
            let (nd_view, rough_view) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views()[i], gb.roughness_views()[i]),
                None => (self.ssao_white.view, self.ssao_white.view),
            };
            write_composite_channel_set(
                &self.device,
                set,
                nd_view,
                rough_view,
                self.transient_pool
                    .view_for("ao_output", i)
                    .unwrap_or(self.ssao_white.view),
                self.composite.sampler.handle(),
            );
        }

        // The render-finished semaphores are one-per-swapchain-image; a
        // resize can change the image count, so resize the pool to match.
        // wait_idle() above guarantees none are still in flight.
        if self.frame_sync.render_finished.len() != self.swapchain.images.len() {
            for &s in &self.frame_sync.render_finished {
                // SAFETY: the handle was created from this device and is destroyed exactly once;
                // the caller has already waited for the device to go idle, so no submission still
                // references it.
                unsafe { self.device.destroy_semaphore(s, None) };
            }
            let sem_info = vk::SemaphoreCreateInfo::default();
            self.frame_sync.render_finished = (0..self.swapchain.images.len())
                // SAFETY: the create-info and every slice it borrows are live for the call, and
                // each handle it names belongs to this device.
                .map(|_| unsafe { self.device.create_semaphore(&sem_info, None) })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("semaphore: {e}"))?;
        }
        Ok(())
    }
}

// Vulkan handles needed to query the surface and create a swapchain against it.
pub(super) struct SwapchainSurface<'a> {
    pub instance: &'a ash::Instance,
    pub device: &'a VkDevice,
    pub pd: vk::PhysicalDevice,
    pub surface_loader: &'a ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub swapchain_loader: &'a ash::khr::swapchain::Device,
}

// Graphics + present queue family indices. Equal indices select EXCLUSIVE
// sharing; distinct indices select CONCURRENT sharing across both families.
#[derive(Clone, Copy)]
pub(super) struct SwapchainQueueFamilies {
    pub graphics_family: u32,
    pub present_family: u32,
}

// Swapchain sizing + presentation configuration.
#[derive(Clone, Copy)]
pub(super) struct SwapchainConfig {
    pub width: u32,
    pub height: u32,
    pub old_swapchain: vk::SwapchainKHR,
    // Resolved output mode, picking the swapchain (format, colour space):
    //   - `Sdr`                        -> `B8G8R8A8_UNORM` + sRGB-nonlinear.
    //   - `Hdr{ ExtendedLinear }`      -> `R16G16B16A16_SFLOAT` +
    //     `EXTENDED_SRGB_LINEAR_EXT` (scRGB linear).
    //   - `Hdr{ Pq }`                  -> HDR10 PQ: `R16G16B16A16_SFLOAT` +
    //     `HDR10_ST2084_EXT` preferred (keeps the composite + screenshot paths
    //     identical to the scRGB float swapchain), else
    //     `A2B10G10R10_UNORM_PACK32` + `HDR10_ST2084_EXT`.
    // The caller has already enabled `VK_EXT_swapchain_colorspace` and gated the
    // resolved mode on the surface advertising the matching pair (see the HDR
    // resolve in init.rs), so the chosen encoding and colour space stay in
    // sync. Each arm falls back through scRGB to the SDR default if its
    // preferred pair is unexpectedly absent.
    pub hdr_mode: crate::gfx::hdr_output::HdrOutputMode,
    // Lock presentation to the display refresh. `true` forces FIFO (always
    // present, vsync); `false` prefers MAILBOX (uncapped render loop, no
    // tearing), then IMMEDIATE, falling back to FIFO when neither is offered.
    pub vsync: bool,
}

// The extent a swapchain will be created at. The surface's own current extent
// wins when it reports one; `u32::MAX` is the spec's "no preference" sentinel,
// and only then does the requested window size decide, clamped to the range the
// surface supports. Windows always reports a real extent, which is what makes
// this (not the window's cached client size) the authority on whether a
// swapchain can be built at all: a minimised window reports 0x0 here.
fn resolve_swapchain_extent(
    caps: &vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        return caps.current_extent;
    }
    vk::Extent2D {
        width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    }
}

// Whether `extent` can carry a swapchain. Vulkan rejects a zero dimension on
// the swapchain and on every attachment, framebuffer, render area, and viewport
// sized from it.
pub(super) fn extent_is_presentable(extent: vk::Extent2D) -> bool {
    extent.width > 0 && extent.height > 0
}

pub(super) fn create_swapchain_inner(
    surface: &SwapchainSurface,
    families: SwapchainQueueFamilies,
    config: SwapchainConfig,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D), String> {
    let &SwapchainSurface {
        instance: _instance,
        device: _device,
        pd,
        surface_loader,
        surface,
        swapchain_loader,
    } = surface;
    let SwapchainQueueFamilies {
        graphics_family,
        present_family,
    } = families;
    let SwapchainConfig {
        width,
        height,
        old_swapchain,
        hdr_mode,
        vsync,
    } = config;
    use crate::gfx::hdr_output::{HdrEncoding, HdrOutputMode};
    // SAFETY: a property query on a live handle; it only reads.
    let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(pd, surface) }
        .map_err(|e| format!("surface caps: {e}"))?;
    // SAFETY: a property query on a live handle; it only reads.
    let formats = unsafe { surface_loader.get_physical_device_surface_formats(pd, surface) }
        .map_err(|e| format!("surface formats: {e}"))?;
    let present_modes =
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { surface_loader.get_physical_device_surface_present_modes(pd, surface) }
            .map_err(|e| format!("present modes: {e}"))?;

    // Pick surface format. scRGB HDR: `R16G16B16A16_SFLOAT` + scRGB-linear
    // (Rec.709 primaries, gamma 1.0, extended range; `1.0` = SDR reference
    // white). HDR10 PQ: a `HDR10_ST2084_EXT` pair (float preferred). SDR:
    // `B8G8R8A8_UNORM` + sRGB-nonlinear. When the preferred pair is absent the
    // arm falls back through scRGB to the first reported format.
    let scrgb_pair = (
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
    );
    let sdr_pair = (
        vk::Format::B8G8R8A8_UNORM,
        vk::ColorSpaceKHR::SRGB_NONLINEAR,
    );
    // PQ candidates, float first so the composite render pass + screenshot
    // read-back stay on the same `R16G16B16A16_SFLOAT` swapchain the scRGB
    // path uses; the 10-bit packed format is the secondary option.
    let pq_pairs = [
        (
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ),
        (
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        ),
    ];
    let pick = |target: (vk::Format, vk::ColorSpaceKHR)| {
        formats
            .iter()
            .find(|f| f.format == target.0 && f.color_space == target.1)
            .copied()
    };
    let surface_format = match hdr_mode {
        HdrOutputMode::Hdr {
            encoding: HdrEncoding::Pq,
            ..
        } => pq_pairs
            .iter()
            .find_map(|&p| pick(p))
            .or_else(|| pick(scrgb_pair))
            .or_else(|| pick(sdr_pair))
            .unwrap_or(formats[0]),
        HdrOutputMode::Hdr { .. } => pick(scrgb_pair)
            .or_else(|| pick(sdr_pair))
            .unwrap_or(formats[0]),
        HdrOutputMode::Sdr => pick(sdr_pair).unwrap_or(formats[0]),
    };

    // FIFO is always available and is the vsync mode. Uncapped prefers MAILBOX
    // (no tearing) then IMMEDIATE (tearing) before falling back to FIFO.
    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        let has = |m: vk::PresentModeKHR| present_modes.contains(&m);
        if has(vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if has(vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };

    let extent = resolve_swapchain_extent(&caps, width, height);

    let image_count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
        u32::MAX
    } else {
        caps.max_image_count
    });

    let queue_families = [graphics_family, present_family];
    let (sharing, families) = if graphics_family == present_family {
        (vk::SharingMode::EXCLUSIVE, &queue_families[..0])
    } else {
        (vk::SharingMode::CONCURRENT, &queue_families[..])
    };

    let sc_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        // TRANSFER_SRC so the `screenshot` debug command can copy the presented
        // image back to a host buffer (see vulkan/screenshot.rs).
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .image_sharing_mode(sharing)
        .queue_family_indices(families)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let swapchain = unsafe { swapchain_loader.create_swapchain(&sc_info, None) }
        .map_err(|e| format!("create swapchain: {e}"))?;

    if old_swapchain != vk::SwapchainKHR::null() {
        // SAFETY: `old_swapchain` was created from this device and was retired into the new
        // swapchain's create-info, which is what makes destroying it here legal; it is destroyed
        // exactly once.
        unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None) };
    }

    // SAFETY: a property query on a live handle; it only reads.
    let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
        .map_err(|e| format!("get swapchain images: {e}"))?;

    Ok((swapchain, images, surface_format.format, extent))
}

pub(super) fn create_swapchain_image_views(
    device: &VkDevice,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>, String> {
    images
        .iter()
        .map(|&img| create_image_view(device, img, format, vk::ImageAspectFlags::COLOR))
        .collect()
}

// Main scene render pass. Renders linear-light HDR into an off-screen
// `R16G16B16A16_SFLOAT` target (the MSAA colour image when multisampled, or
// the resolve image directly otherwise) and ends with the resolve image in
// `SHADER_READ_ONLY_OPTIMAL` so the composite pass can sample it.

// Vulkan handles needed to allocate + transition off-screen attachment images.
pub(super) struct AttachmentDeviceCtx<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// Per-frame color, depth, and HDR-resolve images (one entry per swapchain frame).
type FrameAttachments = (Vec<GpuImage>, Vec<GpuImage>, Vec<GpuImage>);

pub(super) fn create_attachments(
    ctx: &AttachmentDeviceCtx,
    width: u32,
    height: u32,
    msaa: vk::SampleCountFlags,
    count: usize,
) -> Result<FrameAttachments, String> {
    let &AttachmentDeviceCtx {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let upload_ctx = GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    };
    let mut color_images = Vec::new();
    let mut depth_images = Vec::new();
    let mut resolve_images = Vec::new();
    for _ in 0..count {
        let depth = create_depth_image(&upload_ctx, width, height, msaa)?;
        depth_images.push(depth);
        resolve_images.push(create_hdr_resolve_image(
            alloc, device, width, height, HDR_FORMAT,
        )?);
        if msaa != vk::SampleCountFlags::TYPE_1 {
            let color = create_msaa_color_image(&upload_ctx, width, height, HDR_FORMAT, msaa)?;
            color_images.push(color);
        }
    }
    Ok((color_images, depth_images, resolve_images))
}

// Main-pass framebuffers, one per frame-in-flight slot. Each attaches the HDR
// colour (MSAA colour + resolve, or just the resolve image) and depth.
pub(super) fn create_main_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    color_images: &[GpuImage],
    depth_images: &[GpuImage],
    resolve_images: &[GpuImage],
    extent: vk::Extent2D,
    msaa: vk::SampleCountFlags,
) -> Result<Vec<OwnedFramebuffer>, String> {
    (0..resolve_images.len())
        .map(|i| {
            let attachments: Vec<vk::ImageView> = if msaa != vk::SampleCountFlags::TYPE_1 {
                vec![
                    color_images[i].view,
                    depth_images[i].view,
                    resolve_images[i].view,
                ]
            } else {
                vec![resolve_images[i].view, depth_images[i].view]
            };
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            device
                .create_framebuffer(&fb_info)
                .map_err(|e| format!("framebuffer[{i}]: {e}"))
        })
        .collect()
}

// Composite-pass framebuffers, one per swapchain image.
pub(super) fn create_composite_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    swapchain_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<OwnedFramebuffer>, String> {
    swapchain_views
        .iter()
        .enumerate()
        .map(|(i, &sc_view)| {
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(std::slice::from_ref(&sc_view))
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            device
                .create_framebuffer(&fb_info)
                .map_err(|e| format!("composite framebuffer[{i}]: {e}"))
        })
        .collect()
}

// Write a composite descriptor set: binding 0 = HDR resolve image,
// binding 1 = bloom mip 0, binding 2 = the 3D colour-grading LUT. All sampled
// through `sampler`.
pub(super) fn write_composite_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    hdr_view: vk::ImageView,
    bloom_view: vk::ImageView,
    lut_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let hdr_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(hdr_view)
        .sampler(sampler);
    let bloom_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(bloom_view)
        .sampler(sampler);
    let lut_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(lut_view)
        .sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&hdr_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&bloom_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&lut_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Write the composite set's G-buffer channel bindings: normal+depth at 3,
// roughness at 4, the blurred SSAO occlusion at 5. Only the debug view modes
// sample them, but the fragment references all three, so every set is bound
// (the 1x1 white fallback stands in wherever a source does not exist). Split
// from `write_composite_set` because these three survive the scene-input
// re-points TAA / FSR / reflections make.
pub(super) fn write_composite_channel_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    normal_depth_view: vk::ImageView,
    roughness_view: vk::ImageView,
    ao_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let infos = [normal_depth_view, roughness_view, ao_view].map(|view| {
        vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
    });
    let writes: Vec<_> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3 + i as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Create one framebuffer per cascade slice of the array shadow map. Each
// framebuffer attaches a single-layer depth view from
// `shadow_map.aux_views`. Returns one framebuffer per available slice.
pub(super) fn create_shadow_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    shadow_map: &GpuImage,
    size: u32,
) -> Result<Vec<OwnedFramebuffer>, String> {
    let mut fbs = Vec::with_capacity(shadow_map.aux_views.len());
    for &view in &shadow_map.aux_views {
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(size)
            .height(size)
            .layers(1);
        let fb = device
            .create_framebuffer(&fb_info)
            .map_err(|e| format!("shadow framebuffer: {e}"))?;
        fbs.push(fb);
    }
    Ok(fbs)
}

#[cfg(test)]
mod tests {
    use super::{extent_is_presentable, resolve_swapchain_extent};
    use ash::vk;

    fn caps(current: vk::Extent2D) -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            current_extent: current,
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        }
    }

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn a_surface_that_reports_an_extent_decides_the_size() {
        // The requested window size gets no vote, which is the whole point of
        // the minimise gate: the surface knows first.
        let resolved = resolve_swapchain_extent(&caps(extent(800, 600)), 1920, 1080);
        assert_eq!(resolved, extent(800, 600));
    }

    #[test]
    fn the_window_size_decides_under_the_no_preference_sentinel() {
        let no_preference = caps(extent(u32::MAX, u32::MAX));
        assert_eq!(
            resolve_swapchain_extent(&no_preference, 1280, 720),
            extent(1280, 720)
        );
    }

    #[test]
    fn a_window_size_outside_the_surface_range_clamps_to_it() {
        let no_preference = caps(extent(u32::MAX, u32::MAX));
        assert_eq!(
            resolve_swapchain_extent(&no_preference, 99_999, 0),
            extent(4096, 1)
        );
    }

    // A minimised window collapses the surface to 0x0 while the window itself
    // can still report its pre-minimise size for another frame. Resolving from
    // the window there is what built a whole 0x0 attachment chain.
    #[test]
    fn a_minimised_surface_resolves_to_an_unpresentable_extent() {
        let resolved = resolve_swapchain_extent(&caps(extent(0, 0)), 1024, 768);
        assert_eq!(resolved, extent(0, 0));
        assert!(!extent_is_presentable(resolved));
    }

    #[test]
    fn presentable_requires_both_dimensions() {
        assert!(extent_is_presentable(extent(1, 1)));
        assert!(!extent_is_presentable(extent(0, 720)));
        assert!(!extent_is_presentable(extent(1280, 0)));
        assert!(!extent_is_presentable(extent(0, 0)));
    }
}

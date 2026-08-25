// src/vulkan/quality.rs
//
// Runtime application of the Quality-group settings (TAA / SSAO / SSR / SSGI /
// auto-exposure). Each gates a render pass whose GPU resources (pipelines,
// render targets, descriptor sets) are built once at init from the world's
// PostProcessConfig, so applying a change at runtime means building or tearing
// down those resources, not flipping a uniform.
//
// The reconcile below brings each feature's `Option` field to the desired state
// (constructing a turning-on feature with the same `*Resources::new` the init
// path runs, tearing down a turning-off one), then defers the whole target
// rebuild + descriptor rewire to `rebuild_swapchain` -- the exact path a window
// resize takes. Reusing it means a live toggle produces resources rewired
// identically to a launch with the same config, with no second copy of the
// intricate per-reader rewiring to drift. Bloom, decals, fog, particles, and
// the uploaded geometry are untouched.
//
// Ray-traced reflections toggle the same way, with two extra costs: turning
// them on builds the scene acceleration structure (`build_rt_accel`, a one-shot
// fence-waited BLAS + TLAS over current geometry) plus the inline-`rayQueryEXT`
// reflection pass, so a live enable hitches once proportional to triangle count.
// And RT is only live-toggleable when the device is RT-capable -- the ray-query
// device extensions are enabled at creation whenever capable (see
// `create_logical_device`), since an extension cannot be added later; on an
// RT-incapable GPU or under XeSS the toggle no-ops with a warning and RT stays
// whatever it launched as (persisted for the next launch).

use ash::vk;

use crate::gfx::backend::QualitySettings;

use super::context::VkContext;

impl VkContext {
    // Bring the toggle-controlled features to match `q`, applied between frames
    // (the GraphicsSystem reads the SettingCommand before the next draw_frame).
    // A build failure logs and leaves the prior state intact.
    pub(crate) fn apply_quality_settings(&mut self, q: QualitySettings) {
        if let Err(e) = self.apply_quality_settings_inner(q) {
            tracing::error!("apply_quality_settings: rebuild failed: {e}");
        }
    }

    fn apply_quality_settings_inner(&mut self, q: QualitySettings) -> Result<(), String> {
        // Every teardown / rebuild below frees or replaces GPU resources a prior
        // frame may still reference; drain the device first so the swap is safe.
        self.wait_idle();

        // Desired enabled state per feature, from the resolved QualitySettings.
        // RT is additionally gated on the device being RT-capable: a non-capable
        // device (or XeSS) did not enable the ray-query extensions at creation,
        // so it cannot build the acceleration structure at runtime -- the toggle
        // no-ops with a warning and RT stays whatever it launched as.
        let desired_rt = q.rt_reflections.is_some() && self.rt_capable;
        if q.rt_reflections.is_some() && !self.rt_capable {
            tracing::warn!(
                "ray-traced reflections requested but the device is not RT-capable \
                 (no ray-query extensions / XeSS active); keeping SSR"
            );
        }
        let desired_ssr = q.ssr.is_some();
        let desired_ssgi = q.ssgi.is_some();
        let desired_ssao = q.ssao.is_some();
        let desired_ae = q.auto_exposure.is_some();
        // TAA resources are forced present while temporal upscaling is active
        // (the upscaler consumes the velocity pre-pass); the TAA resolve is then
        // dropped from the graph. Mirrors the init `taa_enabled` derivation.
        let upscale_on = self.upscale.is_some();
        let desired_taa = q.taa || upscale_on;

        // The SSR pre-pass resources (`SsrResources`) exist whenever SSR, SSGI,
        // or RT is on (SSGI and RT reuse the SSR resolve's plumbing); mirrors the
        // init `ssr_opt` gate. The unified G-buffer pre-pass is needed by any
        // screen-space consumer of the merged buffer (RT reads its per-frame
        // normal+depth and roughness).
        let ssr_needed = desired_ssr || desired_ssgi || desired_rt;
        let gbuffer_needed = ssr_needed || desired_ssao || desired_taa;

        let hdr_views: Vec<vk::ImageView> =
            self.hdr_resolve_images.iter().map(|i| i.view).collect();

        // Unified G-buffer pre-pass (shared dependency): build it before any
        // consumer that samples it. Kept alive once built (a later toggle-off of
        // the last consumer leaves it resident until the next launch / resize),
        // which is harmless: with no consumer the graph omits its readers.
        if gbuffer_needed && self.gbuffer.is_none() {
            // Its three colour channels are pool-owned, so the pool has to place
            // them before the pre-pass framebuffers can reference them. Rebuild
            // with the G-buffer gate on first; the `rebuild_swapchain` later in
            // this call rebuilds the pool once more and re-points every reader.
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
                    true,
                    self.render_extent,
                    self.swapchain.extent,
                )?,
            )?;
            let pooled = self.transient_pool.gbuffer_pooled(self.frames_in_flight);
            let gb = super::post::gbuffer::GbufferResources::new(
                super::post::gbuffer::GbufferDeviceCtx {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                super::post::gbuffer::GbufferQueueCtx {
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                super::post::gbuffer::GbufferExtent {
                    width: self.render_extent.width,
                    height: self.render_extent.height,
                    frames: self.frames_in_flight,
                },
                super::post::gbuffer::GbufferSsboLayouts {
                    instance: self.instanced.set_layout.as_ref().map(|l| l.handle()),
                    // Skinned variant is built lazily by `upload_skinned`, as at init.
                    skinned: None,
                },
                self.draw.objects.len(),
                self.hot_reload.enabled,
                &pooled,
            )?;
            self.gbuffer = Some(gb);
        }

        // TAA.
        if desired_taa && self.taa.is_none() {
            let taa = super::post::taa::TaaResources::new(
                &super::post::taa::TaaDeviceContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                self.frames_in_flight,
                self.render_extent,
                &super::post::taa::TaaSceneInputs {
                    hdr_resolve_images: &self.hdr_resolve_images,
                    sampler: self.composite.sampler.handle(),
                },
                self.hot_reload.enabled,
            )?;
            self.taa = Some(taa);
        } else if !desired_taa && self.taa.is_some() {
            let mut taa = self.taa.take().expect("taa present");
            taa.destroy(&self.device);
        }

        // SSR pre-pass + resolve. Built whenever SSR / SSGI / RT is on; a
        // SSGI-only or RT-only build has no authored SSR settings, so fall back
        // to the inert defaults (the resolve never runs, but `new` needs a
        // concrete `SsrSettings`).
        if ssr_needed && self.ssr.is_none() {
            let settings = q
                .ssr
                .unwrap_or_else(|| crate::gfx::ssr::SsrSettings::resolve(0.0, 0.0));
            let ssr = super::post::ssr::SsrResources::new(
                &super::post::ssr::SsrGpuContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                super::post::ssr::SsrExtent {
                    width: self.render_extent.width,
                    height: self.render_extent.height,
                },
                self.frames_in_flight,
                super::post::ssr::SsrInitInputs {
                    settings,
                    hdr_resolve_views: &hdr_views,
                    prefilter_view: self.env_map.prefilter.view,
                    cube_sampler: self.cube_sampler.handle(),
                    global_set_layout: self.descriptors.global_set_layout.handle(),
                    probe_cube_count: self.descriptors.probe_cube_count,
                },
                self.hot_reload.enabled,
            )?;
            self.ssr = Some(ssr);
        } else if !ssr_needed && self.ssr.is_some() {
            let mut ssr = self.ssr.take().expect("ssr present");
            ssr.destroy(&self.device);
        }

        // SSGI (samples the unified G-buffer's per-frame normal+depth views).
        if desired_ssgi && self.ssgi.is_none() {
            let settings = q.ssgi.expect("desired_ssgi implies ssgi settings");
            let nd_views = self
                .gbuffer
                .as_ref()
                .expect("SSGI requires the unified G-buffer pre-pass")
                .normal_depth_views();
            let ssgi = super::post::ssgi::SsgiResources::new(
                super::post::ssgi::SsgiDevice {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                self.render_extent.width,
                self.render_extent.height,
                self.frames_in_flight,
                settings,
                super::post::ssgi::SsgiInputViews {
                    hdr_resolve_views: &hdr_views,
                    gbuffer_view: nd_views[0],
                },
                self.hot_reload.enabled,
            )?;
            self.ssgi = Some(ssgi);
        } else if !desired_ssgi && self.ssgi.is_some() {
            let mut ssgi = self.ssgi.take().expect("ssgi present");
            ssgi.destroy(&self.device);
        }

        // Auto-exposure. When it turns off the static authored EV drives exposure
        // again (the GraphicsSystem re-pushes `update_post_process` after this
        // call), so only the GPU state is swapped here.
        if desired_ae && self.auto_exposure.resources.is_none() {
            let settings = q
                .auto_exposure
                .as_ref()
                .expect("desired_ae implies auto-exposure settings");
            let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
                &self.alloc,
                &self.device,
                self.frames_in_flight,
                &hdr_views,
                self.linear_sampler.handle(),
                self.hot_reload.enabled,
            )?;
            self.auto_exposure.resources = Some(resources);
            self.auto_exposure.state =
                Some(crate::gfx::auto_exposure::AutoExposureState::new(settings));
            self.auto_exposure.settings = q.auto_exposure;
            self.auto_exposure.bias_ev = q.auto_exposure_bias_ev;
        } else if !desired_ae && self.auto_exposure.resources.is_some() {
            let mut ae = self
                .auto_exposure
                .resources
                .take()
                .expect("auto-exposure present");
            ae.destroy(&self.device);
            self.auto_exposure.settings = None;
            self.auto_exposure.state = None;
        }

        // SSAO. Its occlusion target is the transient pool's per-frame
        // `ao_output`, which only exists while SSAO is on, so turning it on means
        // rebuilding the pool (to add `ao_output`) before constructing the SSAO
        // resources. `rebuild_swapchain` below rebuilds the pool again from the
        // now-Some `self.ssao`, then re-points binding 6 at the rebuilt views.
        if desired_ssao && self.ssao.is_none() {
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
                    true,
                    self.post_process.bloom_intensity > 0.0,
                    self.gbuffer.is_some(),
                    self.render_extent,
                    self.swapchain.extent,
                )?,
            )?;
            let settings = q.ssao.expect("desired_ssao implies ssao settings");
            let ao_views = self
                .transient_pool
                .views_for_frames("ao_output", self.frames_in_flight);
            let ssao = super::post::ssao::SsaoResources::new(
                &super::post::ssao::SsaoDeviceCtx {
                    alloc: &self.alloc,
                    device: &self.device,
                },
                self.render_extent.width,
                self.render_extent.height,
                self.frames_in_flight,
                settings,
                &ao_views,
                self.hot_reload.enabled,
            )?;
            self.ssao = Some(ssao);
        } else if !desired_ssao && self.ssao.is_some() {
            let mut ssao = self.ssao.take().expect("ssao present");
            ssao.destroy(&self.device);
        }

        // Ray-traced reflections. Turning on builds the scene acceleration
        // structure (one-shot, fence-waited) + the inline-`rayQueryEXT` pass;
        // turning off tears both down. The G-buffer pre-pass RT samples is
        // already built above (`gbuffer_needed` folds in `desired_rt`).
        // `rebuild_swapchain` below then rebuilds the RT output target + re-points
        // the bloom prefilter / composite scene input at it (or off it on a
        // turn-off); the per-frame TLAS / geometry descriptors are wired by the
        // next `rt_dynamic_update`.
        if desired_rt && self.rt_reflections.is_none() {
            self.build_rt_runtime(q.rt_reflections.expect("desired_rt implies settings"))?;
        } else if !desired_rt && self.rt_reflections.is_some() {
            if let Some(mut rt) = self.rt_reflections.take() {
                rt.destroy(&self.device);
            }
            if let Some(mut accel) = self.rt_accel.take() {
                accel.destroy(&self.device);
            }
        }

        // The SSR *resolve* owns the post-stack scene image only when SSR is
        // authored and RT did not take the slot. Set from the ACTUAL post-build
        // RT state: a failed RT enable falls back to the SSR resolve. Mirrors the
        // init `ssr_resolve_on`.
        self.ssr_resolve_active = desired_ssr && self.rt_reflections.is_none();

        // Reflection composite: present whenever a reflection path owns the scene
        // image. Build it on a turn-on, tear it down on a turn-off; `rebuild_swapchain`
        // below then rebuilds its targets + routes the scene image through its output.
        let reflection_active = self.rt_reflections.is_some() || self.ssr_resolve_active;
        if reflection_active && self.reflection_composite.is_none() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|i| i.view).collect();
            let gb = self
                .gbuffer
                .as_ref()
                .expect("a reflection path forces the unified G-buffer pre-pass");
            let nd_views = gb.normal_depth_views();
            let rough_views = gb.roughness_views();
            let rc = super::post::reflection_composite::ReflectionCompositeResources::new(
                &super::texture::GpuUploadContext {
                    alloc: &self.alloc,
                    device: &self.device,
                    command_pool: self.commands.command_pool,
                    queue: self.graphics_queue,
                },
                self.render_extent.width,
                self.render_extent.height,
                self.frames_in_flight,
                q.reflection_blur_scale,
                &super::post::reflection_composite::CompositeInputViews {
                    hdr_resolve_views: &hdr_views,
                    normal_depth_views: &nd_views,
                    roughness_views: &rough_views,
                },
                self.hot_reload.enabled,
            )?;
            self.reflection_composite = Some(rc);
        } else if !reflection_active && self.reflection_composite.is_some() {
            let mut rc = self.reflection_composite.take().expect("checked is_some");
            rc.destroy(&self.device);
        }

        // Rebuild every target + rewire every reader / the composite chain via
        // the resize path. It rebuilds the transient pool + bloom from the
        // reconciled `self.ssao`, rebuilds each `Some` feature's targets, and
        // re-points the bloom prefilter + composite scene input down the
        // upscale > TAA > reflection-composite > HDR priority chain.
        self.rebuild_swapchain()?;

        // `rebuild_swapchain` only re-points set-0 binding 6 inside its
        // SSAO-present branch, so a turn-off leaves it on the just-destroyed
        // `ao_output`. Point it back at the 1x1 white fallback. (On a turn-on it
        // already moved to the rebuilt `ao_output`, so this is only needed off.)
        if !desired_ssao {
            self.rewire_ssao_white_fallback();
        }
        Ok(())
    }

    // Build the RT acceleration structure + reflection pass at runtime (a live
    // toggle-on). Mirrors the init RT block: an empty scene, an AS-build error,
    // or a shader-compile failure leaves both `rt_accel` / `rt_reflections`
    // `None` and the renderer stays on SSR (a soft failure, returns `Ok`). The
    // caller has ensured the unified G-buffer pre-pass exists and drained the
    // device (`wait_idle`). `rebuild_swapchain` refreshes the output target after.
    fn build_rt_runtime(
        &mut self,
        settings: crate::gfx::rt_reflections::RtReflectionSettings,
    ) -> Result<(), String> {
        let accel = match crate::vulkan::raytrace::build_rt_accel(
            crate::vulkan::raytrace::RtDeviceCtx {
                alloc: &self.alloc,
                instance: &self.instance,
                device: &self.device,
                pd: self.physical_device,
            },
            self.commands.command_pool,
            self.graphics_queue,
            crate::vulkan::raytrace::RtSceneGeometry {
                vertex_buffer: self.geometry.vertex_buffer.buffer(),
                index_buffer: self.geometry.index_buffer.buffer(),
                draw_objects: &self.draw.objects,
                clusters: &self.instanced.clusters,
                albedo_count: self.textures.len(),
                total_vertices: self.rt_static_vertex_count,
                exclude_seethrough: self.seethrough_meshes_enabled(),
            },
            self.frames_in_flight,
            self.hot_reload.enabled,
        ) {
            Ok(Some(accel)) => accel,
            Ok(None) => {
                tracing::info!(
                    "RT reflections enabled but no resident triangle geometry to trace; keeping SSR"
                );
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed (keeping SSR): {e}");
                return Ok(());
            }
        };

        let hdr_views: Vec<vk::ImageView> =
            self.hdr_resolve_images.iter().map(|i| i.view).collect();
        let gb = self
            .gbuffer
            .as_ref()
            .expect("RT enable forces the unified G-buffer pre-pass on");
        let nd_views = gb.normal_depth_views();
        let rough_views = gb.roughness_views();
        let (geom_buffer, geom_size) = accel.geom_table();
        // The textured hit variant indexes the bindless pool, so it compiles
        // against the length the pool set layout was built with; 0 when the
        // legacy per-draw path is active (no bindless layout, so the textured
        // variant is not built).
        let bindless_pool_size = self.cull.bindless_pool_size;
        let rt = match super::post::rt_reflections::RtReflectionsResources::new(
            super::post::rt_reflections::RtBuild {
                alloc: &self.alloc,
                device: &self.device,
                width: self.render_extent.width,
                height: self.render_extent.height,
                frames: self.frames_in_flight,
            },
            settings,
            super::post::rt_reflections::RtStaticInputs {
                vertex_buffer: self.geometry.vertex_buffer.buffer(),
                index_buffer: self.geometry.index_buffer.buffer(),
                hdr_resolve_views: &hdr_views,
                gbuffer_views: &nd_views,
                roughness_views: &rough_views,
                prefilter_view: self.env_map.prefilter.view,
                cube_sampler: self.cube_sampler.handle(),
            },
            super::post::rt_reflections::RtAccelHandles {
                tlas: accel.tlas(),
                geom_buffer,
                geom_size,
                deformed_verts: accel.deformed_verts(),
                skinned_indices: accel.skinned_indices(),
            },
            super::post::rt_reflections::RtLayoutConfig {
                bindless_set_layout: self.cull.bindless_set_layout.as_ref().map(|l| l.handle()),
                global_set_layout: self.descriptors.global_set_layout.handle(),
                probe_cube_count: self.descriptors.probe_cube_count,
                pool_size: bindless_pool_size,
                hot_reload: self.hot_reload.enabled,
            },
        ) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (keeping SSR): {e}");
                let mut accel = accel;
                accel.destroy(&self.device);
                return Ok(());
            }
        };
        self.rt_accel = Some(accel);
        self.rt_reflections = Some(rt);
        Ok(())
    }

    // Point set-0 binding 6 (the SSAO occlusion input) at the per-frame pooled
    // `ao_output` when present, else the 1x1 white fallback, on every global
    // set. Used after a live SSAO toggle-off, where the transient pool no longer
    // holds `ao_output` and the main pass's `ambient *= ao` must collapse to a
    // pass-through 1.0. Mirrors the rewire in `rebuild_swapchain`'s SSAO branch.
    fn rewire_ssao_white_fallback(&self) {
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
    }
}

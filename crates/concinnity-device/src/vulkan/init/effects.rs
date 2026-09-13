//! Post and world effects: the temporal upscaler, the screen-space passes
//! sharing the unified G-buffer, TAA and the scene-input rewiring, and the
//! decal, fog, raymarch, planar reflection, transparent and auto-exposure
//! resources.

use ash::vk;
use concinnity_core::components::{self, GlassPanel, SdfVolume, WaterSurface};
use concinnity_core::gfx::auto_exposure::{self, AutoExposureSettings};
use concinnity_core::gfx::ssao::SsaoSettings;
use concinnity_core::gfx::ssgi::SsgiSettings;
use concinnity_core::gfx::ssr::SsrSettings;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::planar_reflection;
use concinnity_core::render::volumetric_fog::FogSettings;

use super::{GlobalBindings, InitGpu};
use crate::vulkan::allocator::PooledBuffer;
use crate::vulkan::auto_exposure::AutoExposureResources;
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::decal::DecalResources;
use crate::vulkan::fog::FogResources;
use crate::vulkan::hiz::HiZResources;
use crate::vulkan::owned::{OwnedRenderPass, OwnedSampler, OwnedSetLayout};
use crate::vulkan::planar::PlanarReflectionSet;
use crate::vulkan::post::PostSupport;
use crate::vulkan::post::bloom::rebind_bloom_input0;
use crate::vulkan::post::gbuffer::{GbufferPooled, GbufferResources};
use crate::vulkan::post::post_device::{PostQueue, VkPostDevice, VkPostProbes};
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::post::ssao::SsaoResources;
use crate::vulkan::post::ssgi::SsgiResources;
use crate::vulkan::post::ssr::SsrResources;
use crate::vulkan::post::taa::TaaResources;
use crate::vulkan::post::upscale::VkUpscaleBackend;
use crate::vulkan::raymarch::RaymarchResources;
use crate::vulkan::raytrace::RtAccelData;
use crate::vulkan::swapchain::{write_composite_channel_set, write_composite_set};
use crate::vulkan::texture::{self, GpuImage, GpuUploadContext};
use crate::vulkan::transient_pool::TransientImagePool;
use crate::vulkan::transparent::TransparentResources;

pub(super) struct Upscale {
    pub(super) upscale: Option<Box<dyn VkUpscaleBackend>>,
    pub(super) render_extent: vk::Extent2D,
}

// Temporal upscaling (FSR / DLSS / XeSS). Built here, before the
// off-screen attachments, because its render dims drive `render_extent`:
// when an upscaler builds, the whole scene pipeline renders at
// `round(swapchain_extent * upscale_scale)` and the upscaler
// reconstructs the swapchain resolution. When `temporal_upscaling` is
// off (or no backend is available) `render_extent == swapchain_extent`
// and the pipeline collapses to native-resolution rendering. Bloom /
// composite / swapchain always stay at `swapchain_extent`.
// `build_upscaler` resolves `upscale_backend` against availability with
// a DLSS -> XeSS -> FSR -> native fallback (the DLSS / XeSS device
// extensions were enabled at device creation via `upscale_sdk`).
pub(super) fn build_upscale(
    gpu: &InitGpu<'_>,
    swapchain_extent: vk::Extent2D,
    temporal_upscaling: bool,
    upscale_scale: f32,
    upscale_backend: components::UpscalerBackend,
) -> RenderResult<Upscale> {
    let InitGpu {
        instance,
        device,
        physical_device,
        alloc,
        command_pool,
        queue: graphics_queue,
        ..
    } = *gpu;
    let upscale = if temporal_upscaling {
        let (built, resolved) = crate::vulkan::post::build_upscaler(
            crate::vulkan::post::upscale::UpscalerGpu {
                alloc,
                instance,
                device,
                physical_device,
                command_pool,
                queue: graphics_queue,
            },
            swapchain_extent.width,
            swapchain_extent.height,
            upscale_scale,
            upscale_backend,
        )?;
        // Arm the messenger's benign-error budget for DLSS (see
        // `DLSS_FIRST_FRAME_LAYOUT_SUPPRESS`); a no-op for other backends.
        if resolved == crate::vulkan::post::ResolvedBackend::Dlss
            && let Some(f) = device.debug_filter()
        {
            f.store(
                super::bootstrap::DLSS_FIRST_FRAME_LAYOUT_SUPPRESS,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        built
    } else {
        None
    };
    let render_extent = match &upscale {
        Some(u) => {
            let (w, h) = u.render_dims();
            vk::Extent2D {
                width: w,
                height: h,
            }
        }
        None => swapchain_extent,
    };
    Ok(Upscale {
        upscale,
        render_extent,
    })
}

struct PostDeviceSources<'a> {
    post_support: &'a PostSupport,
    composite_sampler: &'a OwnedSampler,
    cube_sampler: &'a OwnedSampler,
    global_set_layout: &'a OwnedSetLayout,
    probe_cube_count: u32,
}

// The device every shared post pass builds its pipelines and targets
// through at init. Nothing encodes before the context exists, so it
// carries the global set's layout but none of its per-frame sets.
fn shared_post_device<'a>(gpu: &InitGpu<'a>, sources: PostDeviceSources<'a>) -> VkPostDevice<'a> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        hot_reload,
        ..
    } = *gpu;
    let PostDeviceSources {
        post_support,
        composite_sampler,
        cube_sampler,
        global_set_layout,
        probe_cube_count,
    } = sources;
    VkPostDevice {
        device,
        alloc,
        queue: PostQueue {
            command_pool,
            queue: graphics_queue,
        },
        cache: &post_support.cache,
        arena: &post_support.arena,
        sampler: composite_sampler.handle(),
        cube_sampler: cube_sampler.handle(),
        probes: Some(VkPostProbes {
            layout: global_set_layout.handle(),
            sets: &[],
            cube_count: probe_cube_count,
        }),
        frame: 0,
        hot_reload,
    }
}

pub(super) struct ScreenSpaceInputs<'a> {
    pub(super) transient_pool: &'a TransientImagePool,
    pub(super) gbuffer_pooled: &'a GbufferPooled,
    pub(super) composite_sampler: &'a OwnedSampler,
    pub(super) cube_sampler: &'a OwnedSampler,
    pub(super) global_set_layout: &'a OwnedSetLayout,
    pub(super) probe_cube_count: u32,
    pub(super) render_extent: vk::Extent2D,
    pub(super) ssao_settings: Option<SsaoSettings>,
    pub(super) ssr_settings: Option<SsrSettings>,
    pub(super) ssgi_settings: Option<SsgiSettings>,
    pub(super) rt_wanted: bool,
    pub(super) gbuffer_enabled: bool,
}

pub(super) struct ScreenSpace {
    pub(super) ssao_white: GpuImage,
    pub(super) ssao_opt: Option<SsaoResources>,
    pub(super) ssr_authored: bool,
    pub(super) post_support: PostSupport,
    pub(super) ssr_opt: Option<SsrResources>,
    pub(super) gbuffer_opt: Option<GbufferResources>,
    pub(super) ssgi_opt: Option<SsgiResources>,
}

pub(super) fn build_screen_space(
    gpu: &InitGpu<'_>,
    inputs: ScreenSpaceInputs<'_>,
) -> RenderResult<ScreenSpace> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let ScreenSpaceInputs {
        transient_pool,
        gbuffer_pooled,
        composite_sampler,
        cube_sampler,
        global_set_layout,
        probe_cube_count,
        render_extent,
        ssao_settings,
        ssr_settings,
        ssgi_settings,
        rt_wanted,
        gbuffer_enabled,
    } = inputs;
    // SSAO (GTAO): pre-pass + kernel + blur, plus a 1×1 white fallback
    // that is always bound at set 0 binding 6 when SSAO is off so the
    // main pass's `ambient *= ao` multiplier collapses to a pass-through.
    let ssao_white = texture::create_fallback_white(&GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue: graphics_queue,
    })?;
    // The transient image pool was built with the render targets (before the bloom chain); it
    // already holds this frame's pooled `ao_output` views when SSAO is on.
    let ssao_opt = if let Some(settings) = ssao_settings {
        let ao_views = transient_pool.views_for_frames("ao_output", frames);
        Some(crate::vulkan::post::ssao::SsaoResources::new(
            &crate::vulkan::post::ssao::SsaoDeviceCtx { alloc, device },
            render_extent.width,
            render_extent.height,
            frames,
            settings,
            &ao_views,
            hot_reload,
        )?)
    } else {
        None
    };

    // SSR (screen-space reflections): depth + normal + roughness pre-pass
    // and a fullscreen ray-march resolve. The pre-pass G-buffer is shared
    // with SSGI and RT, so `SsrResources` is built whenever any of them is
    // on; its settings stay `None` unless SSR itself is authored, and the
    // resolve runs only when it is and RT did not take its graph slot.
    let ssr_authored = ssr_settings.is_some();
    // RT reflections reuse the SSR depth + normal + roughness pre-pass
    // G-buffer (like SSGI), so the pre-pass half is built whenever SSR, SSGI,
    // *or* RT (and the device supports it) is on. `rt_wanted` is derived up
    // with the transient pool's gates.
    let post_support = crate::vulkan::post::PostSupport::new(device, frames)?;
    let init_post_device = shared_post_device(
        gpu,
        PostDeviceSources {
            post_support: &post_support,
            composite_sampler,
            cube_sampler,
            global_set_layout,
            probe_cube_count,
        },
    );
    let ssr_opt = if ssr_settings.is_some() || ssgi_settings.is_some() || rt_wanted {
        Some(crate::vulkan::post::ssr::SsrResources::new(
            &init_post_device,
            ssr_settings,
            render_extent,
        )?)
    } else {
        None
    };

    // Unified geometry G-buffer pre-pass. Built whenever any screen-space
    // consumer of the merged buffer is on: SSR resolve / SSGI / RT (all
    // fold into `ssr_opt`), SSAO, or the velocity channel a TAA / upscale
    // consumer needs (`taa_enabled`). One jittered traversal rasterizes the
    // normal+depth / roughness / velocity MRT every reader then samples,
    // replacing the separate SSR / SSAO / velocity pre-passes. The skinned
    // variant is built lazily by `upload_skinned` once the joint-set layout
    // exists (it doesn't at init). Mirrors the DirectX `self.gbuffer` build.
    // The gate is `gbuffer_enabled`, the same value the transient pool was built
    // from, so the pool cannot place the MRT channels for a pre-pass that is
    // not built (harmless) or -- the dangerous direction -- leave them
    // unplaced for one that is.
    let gbuffer_opt = if gbuffer_enabled {
        Some(crate::vulkan::post::gbuffer::GbufferResources::new(
            crate::vulkan::post::gbuffer::GbufferDeviceCtx { alloc, device },
            crate::vulkan::post::gbuffer::GbufferQueueCtx {
                command_pool,
                queue: graphics_queue,
            },
            crate::vulkan::post::gbuffer::GbufferExtent {
                width: render_extent.width,
                height: render_extent.height,
                frames,
            },
            gbuffer_pooled,
        )?)
    } else {
        None
    };

    // SSGI (screen-space global illumination): the hemisphere-gather +
    // depth-aware-blur GI pass. Built only when the world selected
    // `indirect_lighting: ssgi`; it samples the unified pre-pass G-buffer,
    // which SSGI forces on, and each frame's HDR resolve.
    let ssgi_opt = match ssgi_settings {
        Some(settings) => Some(crate::vulkan::post::ssgi::SsgiResources::new(
            &init_post_device,
            settings,
            render_extent,
        )?),
        None => None,
    };
    Ok(ScreenSpace {
        ssao_white,
        ssao_opt,
        ssr_authored,
        post_support,
        ssr_opt,
        gbuffer_opt,
        ssgi_opt,
    })
}

pub(super) struct SceneInputWiring<'a> {
    pub(super) taa_enabled: bool,
    pub(super) render_extent: vk::Extent2D,
    pub(super) post_support: &'a PostSupport,
    pub(super) composite_sampler: &'a OwnedSampler,
    pub(super) cube_sampler: &'a OwnedSampler,
    pub(super) global_set_layout: &'a OwnedSetLayout,
    pub(super) probe_cube_count: u32,
    pub(super) composite_sets: &'a [vk::DescriptorSet],
    pub(super) bloom_input_sets: &'a [Vec<vk::DescriptorSet>],
    pub(super) bloom_mips: &'a [Vec<GpuImage>],
    pub(super) color_lut: &'a GpuImage,
    pub(super) upscale: Option<&'a dyn VkUpscaleBackend>,
    pub(super) gbuffer_opt: Option<&'a GbufferResources>,
    pub(super) ssao_opt: Option<&'a SsaoResources>,
    pub(super) ssao_white: &'a GpuImage,
    pub(super) transient_pool: &'a TransientImagePool,
}

pub(super) fn build_taa_and_wire_scene_inputs(
    gpu: &InitGpu<'_>,
    inputs: SceneInputWiring<'_>,
) -> RenderResult<Option<TaaResources>> {
    let InitGpu { device, frames, .. } = *gpu;
    let SceneInputWiring {
        taa_enabled,
        render_extent,
        post_support,
        composite_sampler,
        cube_sampler,
        global_set_layout,
        probe_cube_count,
        composite_sets,
        bloom_input_sets,
        bloom_mips,
        color_lut,
        upscale,
        gbuffer_opt,
        ssao_opt,
        ssao_white,
        transient_pool,
    } = inputs;
    let init_post_device = shared_post_device(
        gpu,
        PostDeviceSources {
            post_support,
            composite_sampler,
            cube_sampler,
            global_set_layout,
            probe_cube_count,
        },
    );
    // When TAA is on the history resolve produces a post-TAA scene image;
    // the bloom prefilter and composite pass must sample that instead of the
    // raw HDR resolve, so their binding-0 descriptor is re-pointed at the
    // per-frame TAA output image. The resolve's own inputs need no wiring:
    // it allocates its set per frame from the shared post arena.
    let taa = if taa_enabled {
        let taa = TaaResources::new(&init_post_device, frames, render_extent)?;
        for (i, &set) in composite_sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                taa.output_view(i),
                bloom_mips[i][0].view,
                color_lut.view,
                composite_sampler.handle(),
            );
        }
        for (i, frame_sets) in bloom_input_sets.iter().enumerate() {
            rebind_bloom_input0(
                device,
                frame_sets[0],
                taa.output_view(i),
                composite_sampler.handle(),
            );
        }
        Some(taa)
    } else {
        None
    };

    // Temporal upscaling overrides the scene input: when FSR is active the
    // bloom prefilter + composite sample its reconstructed swapchain-res
    // output (a single shared image), not the per-frame TAA output. TAA
    // resources are forced built under upscaling (for the velocity pre-pass)
    // and the TAA block above pointed the sets at the TAA output, so this
    // override is the final word; the TAA *resolve* is dropped from the
    // graph and never runs.
    if let Some(up) = upscale {
        let up_output_view = up.output_image().view;
        for (i, &set) in composite_sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                up_output_view,
                bloom_mips[i][0].view,
                color_lut.view,
                composite_sampler.handle(),
            );
        }
        for frame_sets in bloom_input_sets {
            rebind_bloom_input0(
                device,
                frame_sets[0],
                up_output_view,
                composite_sampler.handle(),
            );
        }
    }

    // Re-point the SSAO kernel/blur's G-buffer descriptors at the merged
    // pre-pass's per-frame views now that the merged buffer exists. RT was
    // already wired to the unified views at its construction, and the shared
    // post passes read those views per frame, so they need no wiring.
    if let Some(gb) = gbuffer_opt {
        let nd_views = gb.normal_depth_views();
        if let Some(ssao) = ssao_opt {
            ssao.wire_kernel_and_blur_sets_gbuffer(device, &nd_views);
        }
    }

    // Composite G-buffer channel bindings (3/4/5), for the debug view
    // modes. Written after the re-wire above so they point at the merged
    // pre-pass's views; the 1x1 white fallback stands in when a world built
    // no G-buffer / no SSAO.
    for (i, &set) in composite_sets.iter().enumerate() {
        let (nd_view, rough_view) = match gbuffer_opt {
            Some(gb) => (gb.normal_depth_views()[i], gb.roughness_views()[i]),
            None => (ssao_white.view, ssao_white.view),
        };
        write_composite_channel_set(
            device,
            set,
            nd_view,
            rough_view,
            transient_pool
                .view_for("ao_output", i)
                .unwrap_or(ssao_white.view),
            composite_sampler.handle(),
        );
    }
    Ok(taa)
}

pub(super) struct WorldEffectInputs<'a> {
    pub(super) render_extent: vk::Extent2D,
    pub(super) msaa_samples: vk::SampleCountFlags,
    pub(super) depth_images: &'a [GpuImage],
    pub(super) hdr_resolve_images: &'a [GpuImage],
    pub(super) main_render_pass: &'a OwnedRenderPass,
    pub(super) shadow_render_pass: &'a OwnedRenderPass,
    pub(super) global_set_layout: &'a OwnedSetLayout,
    pub(super) global_update_after_bind: bool,
    pub(super) probe_cube_count: u32,
    pub(super) fog_settings: Option<&'a FogSettings>,
    pub(super) sdf_volumes: &'a [(SdfVolume, Vec<u8>, String)],
    pub(super) water_surfaces: &'a [WaterSurface],
    pub(super) glass_panels: &'a [GlassPanel],
    pub(super) planar_planes: usize,
    pub(super) cull_set_layout: Option<&'a OwnedSetLayout>,
    pub(super) object_buffers: &'a [PooledBuffer],
    pub(super) draw_args_buffers: &'a [PooledBuffer],
    pub(super) n_cull: usize,
    pub(super) hiz: Option<&'a HiZResources>,
    pub(super) composite_opt: Option<&'a ReflectionCompositeResources>,
    pub(super) rt_accel_opt: Option<&'a RtAccelData>,
    pub(super) rt_capable: bool,
    pub(super) has_seethrough_meshes: bool,
    pub(super) seethrough_mesh_indices: &'a [usize],
    pub(super) vertex_buffer: &'a PooledBuffer,
    pub(super) index_buffer: &'a PooledBuffer,
    pub(super) bindless_set_layout: Option<&'a OwnedSetLayout>,
    pub(super) bindless_pool_size: usize,
    pub(super) auto_exposure_settings: Option<&'a AutoExposureSettings>,
}

pub(super) struct WorldEffects {
    pub(super) decals_state: Option<DecalResources>,
    pub(super) fog_resources: Option<FogResources>,
    pub(super) raymarch: Option<RaymarchResources>,
    pub(super) planar_reflection: Option<PlanarReflectionSet>,
    pub(super) transparent: Option<TransparentResources>,
    pub(super) auto_exposure: Option<AutoExposureResources>,
    pub(super) auto_exposure_state: Option<auto_exposure::AutoExposureState>,
}

pub(super) fn build_world_effects(
    gpu: &InitGpu<'_>,
    inputs: WorldEffectInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<WorldEffects> {
    let InitGpu {
        instance,
        device,
        physical_device,
        alloc,
        command_pool,
        queue: graphics_queue,
        frames,
        hot_reload,
    } = *gpu;
    let WorldEffectInputs {
        render_extent,
        msaa_samples,
        depth_images,
        hdr_resolve_images,
        main_render_pass,
        shadow_render_pass,
        global_set_layout,
        global_update_after_bind,
        probe_cube_count,
        fog_settings,
        sdf_volumes,
        water_surfaces,
        glass_panels,
        planar_planes,
        cull_set_layout,
        object_buffers,
        draw_args_buffers,
        n_cull,
        hiz,
        composite_opt,
        rt_accel_opt,
        rt_capable,
        has_seethrough_meshes,
        seethrough_mesh_indices,
        vertex_buffer,
        index_buffer,
        bindless_set_layout,
        bindless_pool_size,
        auto_exposure_settings,
    } = inputs;
    let GlobalBindings {
        light_ubo_buffers,
        light_ubo_size,
        shadow_ubos,
        shadow_ubo_size,
        local_light_buffer,
        local_light_buffer_size,
        light_cull,
        shadow_map,
        shadow_sampler,
        spot_shadow,
        area_light_buffer,
        ltc_matrix_image,
        ltc_magnitude_image,
        ltc_sampler,
        env_map,
        cube_sampler,
        ssao_white,
        linear_sampler,
        ..
    } = *bindings;
    // Pipeline + per-frame uniforms + per-decal albedo sets are always
    // built so runtime `add_decal` works from a world that started
    // with none. The encoder simply skips when every slot is `None`
    // or every live decal culls.
    let depth_views: Vec<vk::ImageView> = depth_images.iter().map(|img| img.view).collect();
    let hdr_resolve_views: Vec<vk::ImageView> =
        hdr_resolve_images.iter().map(|img| img.view).collect();
    let decals_state = Some(crate::vulkan::decal::DecalResources::new(
        crate::vulkan::decal::DecalDeviceContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        crate::vulkan::decal::DecalPassTargets {
            hdr_format: HDR_FORMAT,
            hdr_resolve_views: &hdr_resolve_views,
            depth_views: &depth_views,
            sampler: linear_sampler.handle(),
            extent: render_extent,
        },
        frames,
        msaa_samples != vk::SampleCountFlags::TYPE_1,
        hot_reload,
    )?);

    // Volumetric fog: pipeline + per-frame uniform ring. Built only
    // when the world declared a `VolumetricFog`; the encoder skips the
    // pass when `fog_settings` is `None`.
    let fog_resources = if fog_settings.is_some() {
        Some(crate::vulkan::fog::FogResources::new(
            crate::vulkan::fog::FogDeviceContext {
                alloc,
                device,
                command_pool,
                queue: graphics_queue,
            },
            crate::vulkan::fog::FogFrameTargets {
                frames,
                msaa: msaa_samples != vk::SampleCountFlags::TYPE_1,
                hdr_format: HDR_FORMAT,
                hdr_resolve_views: &hdr_resolve_views,
                depth_views: &depth_views,
                sampler: linear_sampler.handle(),
                extent: render_extent,
            },
            crate::vulkan::fog::FogShadowResources {
                ubos: shadow_ubos,
                map_view: shadow_map.view,
                sampler: shadow_sampler.handle(),
            },
            hot_reload,
        )?)
    } else {
        None
    };

    // Raymarched SDF volumes: per-volume pipelines + the shared view ring,
    // descriptor pool, render passes, and scene snapshot. `None` when no
    // `.glsl` `SdfVolume` survived the backend filter, so the Raymarch pass
    // is omitted from the frame graph.
    let raymarch = crate::vulkan::raymarch::RaymarchResources::try_new(
        crate::vulkan::raymarch::RaymarchDeviceContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        crate::vulkan::raymarch::RaymarchTargetConfig {
            frames,
            msaa_samples,
            width: render_extent.width,
            height: render_extent.height,
        },
        crate::vulkan::raymarch::RaymarchSharedBindings {
            shadow_map_view: shadow_map.view,
            shadow_sampler: shadow_sampler.handle(),
            irradiance_view: env_map.irradiance.view,
            prefilter_view: env_map.prefilter.view,
            cube_sampler: cube_sampler.handle(),
            linear_sampler: linear_sampler.handle(),
            light_ubos: light_ubo_buffers,
            shadow_ubos,
            shadow_render_pass: shadow_render_pass.handle(),
        },
        sdf_volumes,
        hot_reload,
    )?;

    // Planar reflections: group each transparent reflector's world-space plane
    // into a bounded set of distinct planes (near-coplanar reflectors share one
    // mirror render; reflectors past the budget fall back to the probe cube),
    // then build one render-resolution mirror target per distinct plane. Built
    // before the transparent pass so each record's planar binding can point at
    // its plane's target. `slots[i]` is reflector `i`'s target slot (or `None`).
    //
    // Water first, then glass, matching the Metal backend, so the two slot
    // ranges are the leading `water_surfaces.len()` entries and the rest.
    let planar_reflectors: Vec<[f32; 4]> = water_surfaces
        .iter()
        // A water surface's rest plane: horizontal at the surface base height.
        .map(|s| [0.0, 1.0, 0.0, -s.center[1]])
        .chain(
            glass_panels
                .iter()
                .map(|p| crate::vulkan::planar::pane_plane(p.normal, p.center)),
        )
        .collect();
    // Cap at the capacity ceiling the reserved planar targets are sized to, so a
    // stale/over-large preset value can never over-allocate.
    let planar_budget = planar_planes.min(crate::vulkan::planar::MAX_PLANAR_PLANES);
    let planar_assignment =
        planar_reflection::assign_planar_slots(&planar_reflectors, planar_budget);
    // The reflected-frustum mirror cull is bindless-only (it needs the GPU cull
    // set layout + the per-frame object/draw-args SSBOs); a non-bindless world
    // has no `cull_set_layout`, so planar is skipped and its panes keep the
    // probe / sky reflection. Mirrors `metal::planar`'s bindless gate.
    let planar_reflection = if planar_assignment.representatives.is_empty() {
        None
    } else if let Some(csl) = cull_set_layout {
        let cull_sources = crate::vulkan::planar::PlanarCullSources {
            frame_object_buffers: object_buffers,
            frame_draw_args_buffers: draw_args_buffers,
            cull_set_layout: csl.handle(),
            cull_count: n_cull,
            hiz: hiz.map(|h| {
                let (view, sampler) = h.read_set_sources();
                (h.read_set_layout.handle(), view, sampler)
            }),
        };
        Some(crate::vulkan::planar::PlanarReflectionSet::new(
            crate::vulkan::planar::PlanarDevice { alloc, device },
            crate::vulkan::planar::PlanarConfig {
                frames,
                sample_count: msaa_samples,
                width: render_extent.width,
                height: render_extent.height,
            },
            &planar_assignment.representatives,
            main_render_pass,
            crate::vulkan::planar::PlanarGlobalSet {
                update_after_bind: global_update_after_bind,
                layout: global_set_layout.handle(),
                probe_cube_count,
            },
            crate::vulkan::planar::PlanarLightingBindings {
                light_ubos: light_ubo_buffers,
                light_size: light_ubo_size,
                local_light_buffer: local_light_buffer.buffer(),
                local_light_size: local_light_buffer_size,
                cluster_params_ubo: light_cull.unclustered_buffer.buffer(),
                cluster_list_buffer: light_cull.cluster_buffer.buffer(),
                spot_shadow_map_view: spot_shadow.map.view,
                spot_shadow_data_buffer: spot_shadow.data_buffer.buffer(),
                area_light_buffer: area_light_buffer.buffer(),
                ltc_matrix_view: ltc_matrix_image.view,
                ltc_magnitude_view: ltc_magnitude_image.view,
                ltc_sampler: ltc_sampler.handle(),
                shadow_ubos,
                shadow_size: shadow_ubo_size,
                shadow_map_view: shadow_map.view,
                shadow_sampler: shadow_sampler.handle(),
                irradiance_view: env_map.irradiance.view,
                prefilter_view: env_map.prefilter.view,
                cube_sampler: cube_sampler.handle(),
                ssao_white_view: ssao_white.view,
                linear_sampler: linear_sampler.handle(),
            },
            cull_sources,
        )?)
    } else {
        None
    };
    let planar_target_views: Vec<vk::ImageView> = planar_reflection
        .as_ref()
        .map(|s| (0..s.plane_count()).map(|i| s.target_view(i)).collect())
        .unwrap_or_default();

    // The shared transparent pass and its producers: water surfaces,
    // translucent glass panes, and see-through glass meshes. `Some` only when
    // the world declared at least one of the three; the mesh case additionally
    // needs an RT-capable device, since its producer is ray-traced only and a
    // pane-less, water-less world would otherwise build the whole pass for a
    // producer that cannot exist. The pass blends into the post-reflection
    // scene image (the reflection composite output when a reflection path is
    // active, else this slot's HDR resolve), so the scene target per frame slot
    // is resolved here; the main-depth views feed the fragments' manual
    // occlusion test.
    let transparent =
        if glass_panels.is_empty() && water_surfaces.is_empty() && !has_seethrough_meshes {
            None
        } else {
            let (scene_views, scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0..frames)
                .map(|i| {
                    if let Some(c) = composite_opt {
                        (c.output.view, c.output.image)
                    } else {
                        (hdr_resolve_images[i].view, hdr_resolve_images[i].image)
                    }
                })
                .unzip();
            let transparent_depth_views: Vec<vk::ImageView> =
                depth_images.iter().map(|img| img.view).collect();
            // The initial acceleration-structure handles for the RT path (`None`
            // when RT is off at launch; the per-frame `rt_dynamic_update` fills the
            // ring before the RT path is taken). The RT pipelines themselves are
            // built whenever the device is RT-capable.
            let rt_inputs = rt_accel_opt.map(|a| {
                let (geom_buffer, geom_size) = a.geom_table();
                crate::vulkan::transparent::TransparentRtInputs {
                    tlas: a.tlas(),
                    geom_buffer,
                    geom_size,
                    deformed_verts: a.deformed_verts(),
                    skinned_indices: a.skinned_indices(),
                }
            });
            let (water_planar_slots, glass_planar_slots) =
                planar_assignment.slots.split_at(water_surfaces.len());
            Some(crate::vulkan::transparent::TransparentResources::new(
                crate::vulkan::transparent::TransparentDeviceCtx {
                    alloc,
                    instance,
                    device,
                    physical_device,
                    command_pool,
                    queue: graphics_queue,
                },
                crate::vulkan::transparent::TransparentBuildConfig {
                    frames,
                    msaa_samples,
                    width: render_extent.width,
                    height: render_extent.height,
                    global_set_layout: global_set_layout.handle(),
                    probe_cube_count,
                    hot_reload,
                },
                crate::vulkan::transparent::TransparentSceneTargets {
                    scene_views: &scene_views,
                    scene_images: &scene_images,
                    depth_views: &transparent_depth_views,
                    sampler: linear_sampler.handle(),
                },
                crate::vulkan::transparent::TransparentContent {
                    glass_panels,
                    glass_planar_slots,
                    water_surfaces,
                    water_planar_slots,
                    planar_target_views: &planar_target_views,
                    seethrough_mesh_indices,
                },
                crate::vulkan::transparent::TransparentRtSetup {
                    rt_capable,
                    vertex_buffer: vertex_buffer.buffer(),
                    index_buffer: index_buffer.buffer(),
                    rt_inputs,
                    bindless_set_layout: bindless_set_layout.map(|l| l.handle()),
                    bindless_pool_size,
                },
            )?)
        };

    // Auto-exposure (EV adaptation): histogram + average compute
    // pipelines, the device-local histogram + output buffers, and the
    // per-frame readback ring. Built only when the world's
    // `PostProcessConfig` opted in. With auto-exposure off every
    // field below is None and the static authored EV continues to
    // drive `post_process.exposure` unchanged.
    let (auto_exposure, auto_exposure_state) = if let Some(settings) = auto_exposure_settings {
        let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
            alloc,
            device,
            frames,
            &hdr_resolve_views,
            linear_sampler.handle(),
            hot_reload,
        )?;
        let state = auto_exposure::AutoExposureState::new(settings);
        (Some(resources), Some(state))
    } else {
        (None, None)
    };
    Ok(WorldEffects {
        decals_state,
        fog_resources,
        raymarch,
        planar_reflection,
        transparent,
        auto_exposure,
        auto_exposure_state,
    })
}

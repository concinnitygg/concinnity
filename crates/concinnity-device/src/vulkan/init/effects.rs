//! Post and world effects: the temporal upscaler, the screen-space passes
//! sharing the unified G-buffer, TAA and the scene-input rewiring, and the
//! decal, fog, raymarch, planar reflection, transparent and auto-exposure
//! resources.

use ash::vk;
use concinnity_engine::gfx::auto_exposure;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::init::{PlanarBudget, WorldFx};
use concinnity_engine::render::config::ResolvedGraphics;
use concinnity_engine::render::planar_reflection::PlanarReflectors;

use super::ray_tracing::RtResources;
use super::{Features, InitGpu};
use crate::vulkan::context::{
    AutoExposureState, CompositeState, CullState, DecalState, Descriptors, FogState, Geometry,
    HDR_FORMAT, SceneAssets, Targets,
};
use crate::vulkan::global_set::GlobalBindings;
use crate::vulkan::planar::PlanarReflectionSet;
use crate::vulkan::post::PostSupport;
use crate::vulkan::post::bloom::composite_bloom_view;
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::post::post_device::{PostProbes, PostQueue, VkPostDevice};
use crate::vulkan::post::ssao::SsaoResources;
use crate::vulkan::post::ssgi::SsgiResources;
use crate::vulkan::post::ssr::SsrResources;
use crate::vulkan::post::taa::TaaResources;
use crate::vulkan::post::upscale::UpscaleBackend;
use crate::vulkan::raymarch::RaymarchResources;
use crate::vulkan::swapchain::{
    composite_channels, write_composite_channel_set, write_composite_set,
};
use crate::vulkan::transparent::TransparentResources;

// Temporal upscaling, built before the off-screen attachments because its
// render dims set the render extent; without it the scene renders at the
// swapchain extent. Falls back DLSS -> XeSS -> FSR -> native.
pub(super) fn build_upscale(
    gpu: &InitGpu<'_>,
    swapchain_extent: vk::Extent2D,
    graphics: &ResolvedGraphics,
    requested: &mut crate::upscale_sdk::UpscaleRequest,
) -> RenderResult<(Option<Box<dyn UpscaleBackend>>, vk::Extent2D)> {
    let InitGpu {
        hw, command_pool, ..
    } = *gpu;
    let device = &hw.device;
    let upscale = if let Some(scale) = graphics.upscaler_scale() {
        let (built, resolved) = crate::vulkan::post::build_upscaler(
            crate::vulkan::post::upscale::UpscalerGpu {
                alloc: &hw.alloc,
                instance: &hw.instance,
                device,
                physical_device: hw.physical_device,
                command_pool,
                queue: hw.graphics_queue,
            },
            (swapchain_extent.width, swapchain_extent.height),
            scale,
            requested,
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
            let (w, h) = u.extent().render;
            vk::Extent2D {
                width: w,
                height: h,
            }
        }
        None => swapchain_extent,
    };
    Ok((upscale, render_extent))
}

// The device every shared post pass builds its pipelines and targets
// through at init. Nothing encodes before the context exists, so it
// carries the global set's layout but none of its per-frame sets.
pub(super) fn shared_post_device<'a>(
    gpu: &InitGpu<'a>,
    post_support: &'a PostSupport,
    scene: &'a SceneAssets,
    descriptors: &'a Descriptors,
) -> VkPostDevice<'a> {
    let InitGpu {
        hw,
        command_pool,
        hot_reload,
        ..
    } = *gpu;
    VkPostDevice {
        device: &hw.device,
        alloc: &hw.alloc,
        queue: PostQueue {
            command_pool,
            queue: hw.graphics_queue,
        },
        cache: &post_support.cache,
        arena: &post_support.arena,
        sampler: post_support.sampler.handle(),
        cube_sampler: scene.cube_sampler.handle(),
        probes: PostProbes {
            layout: descriptors.global_set_layout.handle(),
            sets: &[],
        },
        frame: 0,
        hot_reload,
    }
}

#[derive(Clone, Copy)]
pub(super) struct ScreenSpaceInputs<'a> {
    pub(super) graphics: &'a ResolvedGraphics,
    pub(super) features: &'a Features,
    pub(super) targets: &'a Targets,
    pub(super) scene: &'a SceneAssets,
    pub(super) descriptors: &'a Descriptors,
}

pub(super) struct ScreenSpace {
    pub(super) ssao: Option<SsaoResources>,
    pub(super) post: PostSupport,
    pub(super) ssr: Option<SsrResources>,
    pub(super) gbuffer: Option<GbufferResources>,
    pub(super) ssgi: Option<SsgiResources>,
}

pub(super) fn build_screen_space(
    gpu: &InitGpu<'_>,
    inputs: ScreenSpaceInputs<'_>,
) -> RenderResult<ScreenSpace> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let ScreenSpaceInputs {
        graphics,
        features,
        targets,
        scene,
        descriptors,
    } = inputs;
    let render_extent = targets.render_extent;
    let post_support = PostSupport::new(device, frames)?;
    let init_post_device = shared_post_device(gpu, &post_support, scene, descriptors);

    // SSAO (GTAO): the shared depth copy, kernel + blur. Its blurred output is
    // the transient pool's per-frame `ao_output`, read per frame.
    let ssao = match graphics.ssao() {
        Some(settings) => Some(SsaoResources::new(
            &init_post_device,
            settings,
            render_extent,
        )?),
        None => None,
    };

    // SSR. Its pre-pass G-buffer is shared with SSGI and RT, so it is built for any
    // of them; the resolve runs only when SSR is authored and RT did not take it.
    let ssr = if graphics.ssr().is_some() || graphics.ssgi().is_some() || features.rt_wanted {
        Some(SsrResources::new(
            &init_post_device,
            graphics.ssr(),
            render_extent,
        )?)
    } else {
        None
    };

    // The unified G-buffer pre-pass, built for any screen-space consumer. Gated on
    // the same `gbuffer_enabled` the transient pool was built from, so the pool
    // never leaves its channels unplaced for a built pre-pass.
    let gbuffer = if features.gbuffer_enabled {
        Some(GbufferResources::new(
            crate::vulkan::post::gbuffer::GbufferDeviceCtx { alloc, device },
            crate::vulkan::post::gbuffer::GbufferQueueCtx {
                command_pool,
                queue: hw.graphics_queue,
            },
            crate::vulkan::post::gbuffer::GbufferExtent {
                width: render_extent.width,
                height: render_extent.height,
                frames,
            },
            &targets.transient_pool.gbuffer_pooled(frames),
            hot_reload,
        )?)
    } else {
        None
    };

    // SSGI (screen-space global illumination): the Hi-Z traced, temporally
    // accumulated GI pass. Built only when the world selected
    // `indirect_lighting: ssgi`; it samples the unified pre-pass G-buffer,
    // which SSGI forces on, and each frame's HDR resolve.
    let ssgi = match graphics.ssgi() {
        Some(settings) => Some(SsgiResources::new(
            &init_post_device,
            settings,
            render_extent,
        )?),
        None => None,
    };
    Ok(ScreenSpace {
        ssao,
        post: post_support,
        ssr,
        gbuffer,
        ssgi,
    })
}

#[derive(Clone, Copy)]
pub(super) struct SceneInputWiring<'a> {
    pub(super) features: &'a Features,
    pub(super) targets: &'a Targets,
    pub(super) scene: &'a SceneAssets,
    pub(super) descriptors: &'a Descriptors,
    pub(super) composite: &'a CompositeState,
    pub(super) screen: &'a ScreenSpace,
    pub(super) upscale: Option<&'a dyn UpscaleBackend>,
}

pub(super) fn build_taa_and_wire_scene_inputs(
    gpu: &InitGpu<'_>,
    inputs: SceneInputWiring<'_>,
) -> RenderResult<Option<TaaResources>> {
    let InitGpu { hw, frames, .. } = *gpu;
    let device = &hw.device;
    let SceneInputWiring {
        features,
        targets,
        scene,
        descriptors,
        composite,
        screen,
        upscale,
    } = inputs;
    let init_post_device = shared_post_device(gpu, &screen.post, scene, descriptors);
    // With TAA on, the composite samples the per-frame TAA output instead of the
    // raw HDR resolve.
    let taa = if features.taa_enabled {
        let taa = TaaResources::new(&init_post_device, frames, targets.render_extent)?;
        for (i, &set) in composite.sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                taa.output_view(i),
                composite_bloom_view(&targets.transient_pool, scene.ssao_white.view, i),
                scene.color_lut.view,
            );
        }
        Some(taa)
    } else {
        None
    };

    // Upscaling overrides that: the composite samples the upscaler's output, and
    // the TAA resolve is dropped from the graph.
    if let Some(up) = upscale {
        let up_output_view = up.output().image().view;
        for (i, &set) in composite.sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                up_output_view,
                composite_bloom_view(&targets.transient_pool, scene.ssao_white.view, i),
                scene.color_lut.view,
            );
        }
    }

    // Composite channel bindings (3..7), for the debug view modes. Written
    // after the re-wire above so they point at the merged pre-pass's views;
    // the 1x1 white fallback stands in when a world built no G-buffer / no
    // SSAO.
    for (i, &set) in composite.sets.iter().enumerate() {
        let channels = composite_channels(
            targets,
            screen.gbuffer.as_ref(),
            screen.ssao.is_some(),
            scene.ssao_white.view,
            i,
        );
        write_composite_channel_set(device, set, &channels);
    }
    Ok(taa)
}

#[derive(Clone, Copy)]
pub(super) struct WorldEffectInputs<'a> {
    pub(super) targets: &'a Targets,
    pub(super) descriptors: &'a Descriptors,
    pub(super) fx: &'a WorldFx,
    pub(super) planar: PlanarBudget,
    pub(super) cull: &'a CullState,
    pub(super) n_cull: usize,
    pub(super) rt: &'a RtResources,
    pub(super) geometry: &'a Geometry,
    pub(super) graphics: &'a ResolvedGraphics,
}

pub(super) struct WorldEffects {
    pub(super) decal: DecalState,
    pub(super) fog: FogState,
    pub(super) raymarch: Option<RaymarchResources>,
    pub(super) planar_reflection: Option<PlanarReflectionSet>,
    pub(super) transparent: Option<TransparentResources>,
    pub(super) water_sim: Option<crate::vulkan::water_sim::OceanSim>,
    pub(super) auto_exposure: AutoExposureState,
}

use crate::vulkan::texture::GpuUploadContext;

pub(super) fn build_world_effects(
    gpu: &InitGpu<'_>,
    inputs: WorldEffectInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<WorldEffects> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc, graphics_queue) = (&hw.device, &hw.alloc, hw.graphics_queue);
    let WorldEffectInputs {
        targets,
        descriptors,
        fx,
        planar,
        cull,
        n_cull,
        rt,
        geometry,
        graphics,
    } = inputs;
    let GlobalBindings {
        uniforms,
        shadow,
        scene,
        ..
    } = *bindings;
    let (render_extent, msaa_samples) = (targets.render_extent, targets.msaa_samples);
    // Pipeline + per-frame uniforms + per-decal albedo sets are always
    // built so runtime `add_decal` works from a world that started
    // with none. The encoder simply skips when every slot is `None`
    // or every live decal culls.
    let depth_views: Vec<vk::ImageView> = targets.depth_images.iter().map(|img| img.view).collect();
    let hdr_resolve_views: Vec<vk::ImageView> = targets
        .hdr_resolve_images
        .iter()
        .map(|img| img.view)
        .collect();
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
            sampler: scene.linear_sampler.handle(),
            extent: render_extent,
        },
        frames,
        msaa_samples != vk::SampleCountFlags::TYPE_1,
        hot_reload,
    )?);

    // Volumetric fog: pipeline + per-frame uniform ring. Built only
    // when the world declared a `VolumetricFog`; the encoder skips the
    // pass when the scene's fog settings are `None`.
    let fog_resources = if fx.fog.is_some() {
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
                extent: render_extent,
            },
            crate::vulkan::fog::FogShadowResources {
                ubos: &shadow.ubos,
                map_view: shadow.map.view,
                far_view: shadow.far_view(),
                sampler: shadow.sampler.handle(),
            },
            hot_reload,
        )?)
    } else {
        None
    };

    // Raymarched SDF volumes: per-volume pipelines + the shared view ring,
    // descriptor pool, render passes, and scene snapshot. `None` when the world
    // declares no `SdfVolume`, so the Raymarch pass is omitted from the frame graph.
    let raymarch = RaymarchResources::try_new(
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
            shadow_map_view: shadow.map.view,
            far_shadow_view: shadow.far_view(),
            shadow_sampler: shadow.sampler.handle(),
            irradiance_view: scene.env_map.irradiance.view,
            prefilter_view: scene.env_map.prefilter.view,
            cube_sampler: scene.cube_sampler.handle(),
            linear_sampler: scene.linear_sampler.handle(),
            light_ubos: &uniforms.light_ubo_buffers,
            shadow_ubos: &shadow.ubos,
            shadow_render_pass: shadow.render_pass.handle(),
        },
        &fx.sdf_volumes,
        hot_reload,
    )?;

    // Planar reflections: group the reflectors' planes into a bounded set of
    // distinct planes (past the budget, the probe cube), one mirror target each,
    // slots numbered water first. A world with no cull records has nothing to
    // mirror.
    let planar = if n_cull == 0 {
        PlanarBudget {
            planes: 0,
            ..planar
        }
    } else {
        planar
    };
    let planar = PlanarReflectors::plan(&fx.water_surfaces, &fx.glass_panels, planar);
    let planar_slots = planar.assignment.slots.clone();
    // The reflected-frustum mirror cull is bindless-only (it needs the GPU cull
    // set layout + the per-frame object/draw-args SSBOs); a non-bindless world
    // has no `cull_kernels`, so planar is skipped and its panes keep the
    // probe / sky reflection. Mirrors `metal::planar`'s bindless gate.
    let planar_reflection = if planar.planes().is_empty() {
        None
    } else if let (Some(kernels), Some(compact)) =
        (cull.cull_kernels.as_ref(), cull.compact.as_ref())
    {
        let cull_sources = crate::vulkan::planar::PlanarCullSources {
            frame_object_buffers: &cull.object_buffers,
            frame_draw_args_buffers: &cull.draw_args_buffers,
            frame_variant_buffers: &cull.variant_buffers,
            cull_set_layout: kernels.set_layout.handle(),
            compact_set_layout: compact.set_layout.handle(),
            cull_count: n_cull,
            hiz: cull
                .hiz
                .as_ref()
                .map(|h| (h.read_set_layout.handle(), h.read_set_view())),
        };
        Some(PlanarReflectionSet::new(
            crate::vulkan::planar::PlanarDevice {
                alloc,
                device,
                command_pool,
                queue: graphics_queue,
            },
            crate::vulkan::planar::PlanarConfig {
                frames,
                sample_count: msaa_samples,
                width: render_extent.width,
                height: render_extent.height,
            },
            planar,
            &targets.main_render_pass,
            crate::vulkan::planar::PlanarGlobalSet {
                layout: descriptors.global_set_layout.handle(),
                bindings: *bindings,
            },
            cull_sources,
        )?)
    } else {
        None
    };
    let planar_target_views: Vec<vk::ImageView> = planar_reflection
        .as_ref()
        .map_or_default(|s| (0..s.plane_count()).map(|i| s.target_view(i)).collect());

    // The simulated oceans, built before the transparent pass that binds them.
    let water_sim = crate::vulkan::water_sim::OceanSim::for_world(
        &GpuUploadContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        (!fx.water_surfaces.is_empty()).then_some(&fx.ocean),
        rt.reflections.is_some(),
        frames,
        scene.anisotropy,
        hot_reload,
    )?;
    if let Some((reflections, sim)) = rt.reflections.as_ref().zip(water_sim.as_ref()) {
        reflections.wire_waves(device, &sim.views());
    }
    // The transparent pass, when the world declares water, glass panes, or
    // see-through meshes on an RT-capable device. It blends into the
    // post-reflection scene image resolved per frame slot here.
    let transparent = if fx.glass_panels.is_empty()
        && fx.water_surfaces.is_empty()
        && !rt.has_seethrough_meshes
    {
        None
    } else {
        let (scene_views, scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0..frames)
            .map(|i| {
                if let Some(c) = rt.composite.as_ref() {
                    (c.output().view(), c.output().image().image)
                } else {
                    (
                        targets.hdr_resolve_images[i].view,
                        targets.hdr_resolve_images[i].image,
                    )
                }
            })
            .unzip();
        let transparent_depth_views: Vec<vk::ImageView> =
            targets.depth_images.iter().map(|img| img.view).collect();
        let reactive_mask_views: Vec<vk::ImageView> = targets
            .reactive_mask_images
            .iter()
            .map(|img| img.view)
            .collect();
        // The initial acceleration-structure handles for the RT path (`None`
        // when RT is off at launch or there is no BVH yet; the per-frame
        // `rt_dynamic_update` fills the ring before the RT path is taken). The
        // RT pipelines themselves are built whenever the device is RT-capable.
        let rt_inputs = rt.state.accel.as_ref().map(|a| {
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
            planar_slots.split_at(fx.water_surfaces.len());
        Some(TransparentResources::new(
            crate::vulkan::transparent::TransparentDeviceCtx {
                alloc,
                instance: &hw.instance,
                device,
                physical_device: hw.physical_device,
                command_pool,
                queue: graphics_queue,
            },
            crate::vulkan::transparent::TransparentBuildConfig {
                frames,
                msaa_samples,
                width: render_extent.width,
                height: render_extent.height,
                global_set_layout: descriptors.global_set_layout.handle(),
                hot_reload,
                reflection_divisor: graphics.rt_reflections().map_or(1, |rt| rt.divisor),
            },
            crate::vulkan::transparent::TransparentSceneTargets {
                scene_views: &scene_views,
                scene_images: &scene_images,
                reactive_mask_views: &reactive_mask_views,
                depth_views: &transparent_depth_views,
                sampler: scene.linear_sampler.handle(),
            },
            crate::vulkan::transparent::TransparentContent {
                glass_panels: &fx.glass_panels,
                glass_planar_slots,
                water_surfaces: &fx.water_surfaces,
                water_planar_slots,
                ocean: water_sim
                    .as_ref()
                    .map(super::super::water_sim::OceanSim::views),
                planar_target_views: &planar_target_views,
                seethrough_mesh_indices: &rt.seethrough_mesh_indices,
            },
            crate::vulkan::transparent::TransparentRtSetup {
                rt_capable: hw.rt_capable,
                vertex_buffer: geometry.vertex_buffer.buffer(),
                index_buffer: geometry.index_buffer.buffer(),
                rt_inputs,
                bindless_set_layout: cull
                    .bindless_set_layout
                    .as_ref()
                    .map(super::super::owned::OwnedSetLayout::handle),
                bindless_pool_size: cull.bindless_pool_size,
            },
        )?)
    };

    // Auto-exposure, built when `Exposure` opts in; otherwise the authored EV
    // drives the exposure unchanged.
    let (auto_exposure, adaptation) = if let Some(settings) = graphics.auto_exposure().as_ref() {
        let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
            alloc,
            device,
            frames,
            &hdr_resolve_views,
            hot_reload,
        )?;
        let adaptation =
            auto_exposure::ExposureAdaptation::new(*settings, graphics.auto_exposure_bias_ev());
        (Some(resources), Some(adaptation))
    } else {
        (None, None)
    };
    Ok(WorldEffects {
        decal: DecalState {
            resources: decals_state,
        },
        fog: FogState {
            resources: fog_resources,
        },
        raymarch,
        planar_reflection,
        transparent,
        water_sim,
        auto_exposure: AutoExposureState {
            resources: auto_exposure,
            adaptation,
            last_elapsed: 0.0,
        },
    })
}

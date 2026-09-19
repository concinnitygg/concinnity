//! Post and world effects: the temporal upscaler, the screen-space passes
//! sharing the unified G-buffer, TAA and the scene-input rewiring, and the
//! decal, fog, raymarch, planar reflection, transparent and auto-exposure
//! resources.

use ash::vk;
use concinnity_core::gfx::auto_exposure;
use concinnity_core::gfx::render_types::{LightUniforms, ShadowUniforms};
use concinnity_core::render::backend_init::{PostSettings, WorldFx};
use concinnity_core::render::decal;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::lights;
use concinnity_core::render::planar_reflection;

use super::ray_tracing::RtResources;
use super::{Features, GlobalBindings, InitGpu};
use crate::vulkan::context::{
    AutoExposureState, BloomState, CompositeState, DecalState, FogState, HDR_FORMAT, VkCull,
    VkDescriptors, VkGeometry, VkSceneAssets, VkTargets,
};
use crate::vulkan::planar::PlanarReflectionSet;
use crate::vulkan::post::PostSupport;
use crate::vulkan::post::bloom::rebind_bloom_input0;
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::post::post_device::{PostQueue, VkPostDevice, VkPostProbes};
use crate::vulkan::post::ssao::SsaoResources;
use crate::vulkan::post::ssgi::SsgiResources;
use crate::vulkan::post::ssr::SsrResources;
use crate::vulkan::post::taa::TaaResources;
use crate::vulkan::post::upscale::VkUpscaleBackend;
use crate::vulkan::raymarch::RaymarchResources;
use crate::vulkan::swapchain::{write_composite_channel_set, write_composite_set};
use crate::vulkan::transparent::TransparentResources;

// Temporal upscaling (FSR / DLSS / XeSS). Built before the off-screen
// attachments, because its render dims drive the returned render extent:
// when an upscaler builds, the whole scene pipeline renders at
// `round(swapchain_extent * upscale_scale)` and the upscaler
// reconstructs the swapchain resolution. When `temporal_upscaling` is
// off (or no backend is available) the render extent is `swapchain_extent`
// and the pipeline collapses to native-resolution rendering. Bloom /
// composite / swapchain always stay at `swapchain_extent`.
// `build_upscaler` resolves `upscale_backend` against availability with
// a DLSS -> XeSS -> FSR -> native fallback (the DLSS / XeSS device
// extensions were enabled at device creation via `upscale_sdk`).
pub(super) fn build_upscale(
    gpu: &InitGpu<'_>,
    swapchain_extent: vk::Extent2D,
    post: &PostSettings,
) -> RenderResult<(Option<Box<dyn VkUpscaleBackend>>, vk::Extent2D)> {
    let InitGpu {
        hw, command_pool, ..
    } = *gpu;
    let device = &hw.device;
    let upscale = if post.temporal_upscaling {
        let (built, resolved) = crate::vulkan::post::build_upscaler(
            crate::vulkan::post::upscale::UpscalerGpu {
                alloc: &hw.alloc,
                instance: &hw.instance,
                device,
                physical_device: hw.physical_device,
                command_pool,
                queue: hw.graphics_queue,
            },
            swapchain_extent.width,
            swapchain_extent.height,
            post.upscale_scale,
            // Only FSR is available on Vulkan (DLSS / XeSS are DirectX-only); a
            // DX-only request logs a note and uses FSR.
            post.upscale_backend,
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
    Ok((upscale, render_extent))
}

// The device every shared post pass builds its pipelines and targets
// through at init. Nothing encodes before the context exists, so it
// carries the global set's layout but none of its per-frame sets.
fn shared_post_device<'a>(
    gpu: &InitGpu<'a>,
    post_support: &'a PostSupport,
    scene: &'a VkSceneAssets,
    descriptors: &'a VkDescriptors,
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
        probes: Some(VkPostProbes {
            layout: descriptors.global_set_layout.handle(),
            sets: &[],
            cube_count: descriptors.probe_cube_count,
        }),
        frame: 0,
        hot_reload,
    }
}

pub(super) struct ScreenSpaceInputs<'a> {
    pub(super) post: &'a PostSettings,
    pub(super) features: &'a Features,
    pub(super) targets: &'a VkTargets,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) descriptors: &'a VkDescriptors,
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
        post,
        features,
        targets,
        scene,
        descriptors,
    } = inputs;
    let render_extent = targets.render_extent;
    // SSAO (GTAO): pre-pass + kernel + blur. The transient image pool was built
    // with the render targets (before the bloom chain); it already holds this
    // frame's pooled `ao_output` views when SSAO is on.
    let ssao = if let Some(settings) = post.ssao {
        let ao_views = targets.transient_pool.views_for_frames("ao_output", frames);
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
    let post_support = crate::vulkan::post::PostSupport::new(device, frames)?;
    let init_post_device = shared_post_device(gpu, &post_support, scene, descriptors);
    let ssr = if post.ssr.is_some() || post.ssgi.is_some() || features.rt_wanted {
        Some(crate::vulkan::post::ssr::SsrResources::new(
            &init_post_device,
            post.ssr,
            render_extent,
        )?)
    } else {
        None
    };

    // Unified geometry G-buffer pre-pass. Built whenever any screen-space
    // consumer of the merged buffer is on: SSR resolve / SSGI / RT (all
    // fold into `ssr`), SSAO, or the velocity channel a TAA / upscale
    // consumer needs. One jittered traversal rasterizes the normal+depth /
    // roughness / velocity MRT every reader then samples, replacing the
    // separate SSR / SSAO / velocity pre-passes. The skinned variant is built
    // lazily by `upload_skinned` once the joint-set layout exists (it doesn't
    // at init). Mirrors the DirectX `self.gbuffer` build. The gate is the
    // same `gbuffer_enabled` the transient pool was built from, so the pool
    // cannot place the MRT channels for a pre-pass that is not built
    // (harmless) or -- the dangerous direction -- leave them unplaced for one
    // that is.
    let gbuffer = if features.gbuffer_enabled {
        Some(crate::vulkan::post::gbuffer::GbufferResources::new(
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
        )?)
    } else {
        None
    };

    // SSGI (screen-space global illumination): the hemisphere-gather +
    // depth-aware-blur GI pass. Built only when the world selected
    // `indirect_lighting: ssgi`; it samples the unified pre-pass G-buffer,
    // which SSGI forces on, and each frame's HDR resolve.
    let ssgi = match post.ssgi {
        Some(settings) => Some(crate::vulkan::post::ssgi::SsgiResources::new(
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

pub(super) struct SceneInputWiring<'a> {
    pub(super) features: &'a Features,
    pub(super) targets: &'a VkTargets,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) composite: &'a CompositeState,
    pub(super) bloom: &'a BloomState,
    pub(super) screen: &'a ScreenSpace,
    pub(super) upscale: Option<&'a dyn VkUpscaleBackend>,
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
        bloom,
        screen,
        upscale,
    } = inputs;
    let init_post_device = shared_post_device(gpu, &screen.post, scene, descriptors);
    let post_sampler = screen.post.sampler.handle();
    // When TAA is on the history resolve produces a post-TAA scene image;
    // the bloom prefilter and composite pass must sample that instead of the
    // raw HDR resolve, so their binding-0 descriptor is re-pointed at the
    // per-frame TAA output image. The resolve's own inputs need no wiring:
    // it allocates its set per frame from the shared post arena.
    let taa = if features.taa_enabled {
        let taa = TaaResources::new(&init_post_device, frames, targets.render_extent)?;
        for (i, &set) in composite.sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                taa.output_view(i),
                bloom.mips[i][0].view,
                scene.color_lut.view,
                post_sampler,
            );
        }
        for (i, frame_sets) in bloom.input_sets.iter().enumerate() {
            rebind_bloom_input0(device, frame_sets[0], taa.output_view(i), post_sampler);
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
        for (i, &set) in composite.sets.iter().enumerate() {
            write_composite_set(
                device,
                set,
                up_output_view,
                bloom.mips[i][0].view,
                scene.color_lut.view,
                post_sampler,
            );
        }
        for frame_sets in &bloom.input_sets {
            rebind_bloom_input0(device, frame_sets[0], up_output_view, post_sampler);
        }
    }

    // Re-point the SSAO kernel/blur's G-buffer descriptors at the merged
    // pre-pass's per-frame views now that the merged buffer exists. RT was
    // already wired to the unified views at its construction, and the shared
    // post passes read those views per frame, so they need no wiring.
    if let Some(gb) = &screen.gbuffer {
        let nd_views = gb.normal_depth_views();
        if let Some(ssao) = &screen.ssao {
            ssao.wire_kernel_and_blur_sets_gbuffer(device, &nd_views);
        }
    }

    // Composite G-buffer channel bindings (3/4/5), for the debug view
    // modes. Written after the re-wire above so they point at the merged
    // pre-pass's views; the 1x1 white fallback stands in when a world built
    // no G-buffer / no SSAO.
    for (i, &set) in composite.sets.iter().enumerate() {
        let (nd_view, rough_view) = match &screen.gbuffer {
            Some(gb) => (gb.normal_depth_views()[i], gb.roughness_views()[i]),
            None => (scene.ssao_white.view, scene.ssao_white.view),
        };
        write_composite_channel_set(
            device,
            set,
            nd_view,
            rough_view,
            targets
                .transient_pool
                .view_for("ao_output", i)
                .unwrap_or(scene.ssao_white.view),
            post_sampler,
        );
    }
    Ok(taa)
}

pub(super) struct WorldEffectInputs<'a> {
    pub(super) targets: &'a VkTargets,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) fx: &'a WorldFx,
    pub(super) planar_planes: usize,
    pub(super) cull: &'a VkCull,
    pub(super) n_cull: usize,
    pub(super) rt: &'a RtResources,
    pub(super) geometry: &'a VkGeometry,
    pub(super) post: &'a PostSettings,
}

pub(super) struct WorldEffects {
    pub(super) decal: DecalState,
    pub(super) fog: FogState,
    pub(super) raymarch: Option<RaymarchResources>,
    pub(super) planar_reflection: Option<PlanarReflectionSet>,
    pub(super) transparent: Option<TransparentResources>,
    pub(super) auto_exposure: AutoExposureState,
}

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
        planar_planes,
        cull,
        n_cull,
        rt,
        geometry,
        post,
    } = inputs;
    let GlobalBindings {
        uniforms,
        light_cull,
        shadow,
        spot_shadow,
        area_light,
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
    // pass when the fog settings are `None`.
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
                sampler: scene.linear_sampler.handle(),
                extent: render_extent,
            },
            crate::vulkan::fog::FogShadowResources {
                ubos: &shadow.ubos,
                map_view: shadow.map.view,
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
            shadow_map_view: shadow.map.view,
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

    // Planar reflections: group each transparent reflector's world-space plane
    // into a bounded set of distinct planes (near-coplanar reflectors share one
    // mirror render; reflectors past the budget fall back to the probe cube),
    // then build one render-resolution mirror target per distinct plane. Built
    // before the transparent pass so each record's planar binding can point at
    // its plane's target. `slots[i]` is reflector `i`'s target slot (or `None`).
    //
    // Water first, then glass, matching the Metal backend, so the two slot
    // ranges are the leading `water_surfaces.len()` entries and the rest.
    let planar_reflectors: Vec<[f32; 4]> = fx
        .water_surfaces
        .iter()
        // A water surface's rest plane: horizontal at the surface base height.
        .map(|s| [0.0, 1.0, 0.0, -s.center[1]])
        .chain(
            fx.glass_panels
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
    } else if let Some(csl) = cull.cull_set_layout.as_ref() {
        let cull_sources = crate::vulkan::planar::PlanarCullSources {
            frame_object_buffers: &cull.object_buffers,
            frame_draw_args_buffers: &cull.draw_args_buffers,
            cull_set_layout: csl.handle(),
            cull_count: n_cull,
            hiz: cull.hiz.as_ref().map(|h| {
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
            &targets.main_render_pass,
            crate::vulkan::planar::PlanarGlobalSet {
                update_after_bind: descriptors.global_update_after_bind,
                layout: descriptors.global_set_layout.handle(),
                probe_cube_count: descriptors.probe_cube_count,
            },
            crate::vulkan::planar::PlanarLightingBindings {
                light_ubos: &uniforms.light_ubo_buffers,
                light_size: std::mem::size_of::<LightUniforms>() as u64,
                local_light_buffer: uniforms.local_light_buffer.buffer(),
                local_light_size: uniforms.local_light_size,
                cluster_params_ubo: light_cull.unclustered_buffer.buffer(),
                cluster_list_buffer: light_cull.cluster_buffer.buffer(),
                spot_shadow_map_view: spot_shadow.map.view,
                spot_shadow_data_buffer: spot_shadow.data_buffer.buffer(),
                area_light_buffer: area_light.buffer.buffer(),
                ltc_matrix_view: area_light.ltc_matrix.view,
                ltc_magnitude_view: area_light.ltc_magnitude.view,
                ltc_sampler: area_light.sampler.handle(),
                shadow_ubos: &shadow.ubos,
                shadow_size: std::mem::size_of::<ShadowUniforms>() as u64,
                shadow_map_view: shadow.map.view,
                shadow_sampler: shadow.sampler.handle(),
                irradiance_view: scene.env_map.irradiance.view,
                prefilter_view: scene.env_map.prefilter.view,
                cube_sampler: scene.cube_sampler.handle(),
                ssao_white_view: scene.ssao_white.view,
                linear_sampler: scene.linear_sampler.handle(),
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
    let transparent = if fx.glass_panels.is_empty()
        && fx.water_surfaces.is_empty()
        && !rt.has_seethrough_meshes
    {
        None
    } else {
        let (scene_views, scene_images): (Vec<vk::ImageView>, Vec<vk::Image>) = (0..frames)
            .map(|i| {
                if let Some(c) = rt.composite.as_ref() {
                    (c.output.view, c.output.image)
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
        // The initial acceleration-structure handles for the RT path (`None`
        // when RT is off at launch; the per-frame `rt_dynamic_update` fills the
        // ring before the RT path is taken). The RT pipelines themselves are
        // built whenever the device is RT-capable.
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
            planar_assignment.slots.split_at(fx.water_surfaces.len());
        Some(crate::vulkan::transparent::TransparentResources::new(
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
                probe_cube_count: descriptors.probe_cube_count,
                hot_reload,
                reflection_divisor: post.rt_reflections.map_or(1, |rt| rt.divisor),
            },
            crate::vulkan::transparent::TransparentSceneTargets {
                scene_views: &scene_views,
                scene_images: &scene_images,
                depth_views: &transparent_depth_views,
                sampler: scene.linear_sampler.handle(),
            },
            crate::vulkan::transparent::TransparentContent {
                glass_panels: &fx.glass_panels,
                glass_planar_slots,
                water_surfaces: &fx.water_surfaces,
                water_planar_slots,
                planar_target_views: &planar_target_views,
                seethrough_mesh_indices: &rt.seethrough_mesh_indices,
            },
            crate::vulkan::transparent::TransparentRtSetup {
                rt_capable: hw.rt_capable,
                vertex_buffer: geometry.vertex_buffer.buffer(),
                index_buffer: geometry.index_buffer.buffer(),
                rt_inputs,
                bindless_set_layout: cull.bindless_set_layout.as_ref().map(|l| l.handle()),
                bindless_pool_size: cull.bindless_pool_size,
            },
        )?)
    };

    // Auto-exposure (EV adaptation): histogram + average compute
    // pipelines, the device-local histogram + output buffers, and the
    // per-frame readback ring. Built only when the world's
    // `PostProcessConfig` opted in. With auto-exposure off every
    // field below is None and the static authored EV continues to
    // drive `post_process.exposure` unchanged.
    let (auto_exposure, auto_exposure_state) = if let Some(settings) = post.auto_exposure.as_ref() {
        let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
            alloc,
            device,
            frames,
            &hdr_resolve_views,
            scene.linear_sampler.handle(),
            hot_reload,
        )?;
        let state = auto_exposure::AutoExposureState::new(settings);
        (Some(resources), Some(state))
    } else {
        (None, None)
    };
    Ok(WorldEffects {
        decal: DecalState {
            resources: decals_state,
            // Authored decals land in the table through `add_decal`, which
            // also writes each one's albedo descriptor.
            set: decal::DecalSet::new(crate::vulkan::decal::MAX_DECALS, frames),
        },
        // The fog sun mirrors the first directional light, cached because the
        // light UBO is uploaded rather than pushed each frame.
        // `update_directional_lights` re-derives both.
        fog: FogState {
            resources: fog_resources,
            settings: fx.fog,
            sun_dir: lights::sun_direction(&uniforms.light_uniforms),
            sun_color: lights::sun_color(&uniforms.light_uniforms),
        },
        raymarch,
        planar_reflection,
        transparent,
        auto_exposure: AutoExposureState {
            resources: auto_exposure,
            settings: post.auto_exposure,
            state: auto_exposure_state,
            bias_ev: post.auto_exposure_bias_ev,
            last_elapsed: 0.0,
        },
    })
}

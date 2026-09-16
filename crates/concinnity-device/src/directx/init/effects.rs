//! Post and world effects: the temporal upscaler, the fixed descriptor slots of
//! the live-toggleable passes, TAA and the screen-space passes sharing the
//! unified G-buffer, and the decal, fog, particle, auto-exposure, raymarch,
//! planar reflection and transparent resources.

use concinnity_core::components::{GlassPanel, SdfVolume, WaterSurface};
use concinnity_core::gfx::auto_exposure;
use concinnity_core::gfx::render_types::{DrawObject, LightUniforms, NUM_SHADOW_CASCADES};
use concinnity_core::render::backend_init::{PostSettings, WorldFx};
use concinnity_core::render::decal::{self, DecalRecord};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::lights;
use concinnity_core::render::particles::{self, ParticleEmitterRecord};
use concinnity_core::render::planar_reflection::{self, PlanarAssignment};
use concinnity_core::render::post::ssao::SsaoSettings;
use concinnity_core::render::volumetric_fog::FogSettings;
use windows::Win32::Graphics::Direct3D12::*;

use super::heap_layout::{DSV_GBUFFER_DEPTH_SLOT, RtvHeapLayout};
use super::{Features, InitGpu, heaps};
use crate::directx::context::{
    AutoExposureState, DecalState, DxDescriptors, DxSceneAssets, DxTargets, FRAMES, FogState,
    ParticleState, ShadowState, SsaoState, SwapchainState, UpscaleState, dump_on_err,
};
use crate::directx::planar::PlanarReflectionSet;
use crate::directx::post::descriptors::PostDescriptors;
use crate::directx::post::gbuffer::{GbufferResources, GbufferSlots};
use crate::directx::post::post_device::DxPostDevice;
use crate::directx::post::reflection_composite::ReflectionCompositeSlots;
use crate::directx::post::ssao::{SsaoDescriptorHandles, SsaoDeviceCtx, SsaoResources};
use crate::directx::post::ssgi::SsgiResources;
use crate::directx::post::ssr::SsrResources;
use crate::directx::post::taa::TaaResources;
use crate::directx::quality::QualitySlotHandles;
use crate::directx::raymarch::RaymarchResources;
use crate::directx::texture::{create_fallback_white_resource, write_texture_srv};
use crate::directx::transparent::TransparentResources;

// Temporal upscaler (FSR3 via the FidelityFX SDK). Built ahead of the HDR /
// depth / post-effect targets, because its resolved render dimensions decide
// the size of every scene target. When the world's
// `PostProcessConfig.temporal_upscaling` is on AND the FFX DLL loads + the
// context creates successfully, the backend is `Some` and the scene renders at
// `output * upscale_scale`; FSR reconstructs the drawable resolution into the
// upscaler's output texture, which bloom + composite sample. Falls back
// silently (logs a warning, leaves render == output) when the SDK isn't on
// `PATH` or the GPU rejects the context build, so a missing SDK degrades to
// native-resolution TAA rather than a low-res bilinear stretch.
pub(super) fn build_upscale(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    output: (u32, u32),
    post: &PostSettings,
) -> RenderResult<UpscaleState> {
    let hw = gpu.hw;
    let layout = &descriptors.layout;
    let (width, height) = output;
    let upscaler = if post.temporal_upscaling {
        crate::directx::post::upscale::build_upscaler(
            &hw.device,
            &hw.command_queue,
            width,
            height,
            post.upscale_scale,
            crate::directx::post::upscale::UpscalerDescriptors {
                uav_cpu: descriptors.slot_cpu(layout.upscale_uav_slot),
                srv_cpu: descriptors.slot_cpu(layout.upscale_srv_slot),
                srv_gpu: descriptors.slot_gpu(layout.upscale_srv_slot),
            },
            post.upscale_backend,
        )?
        .0
    } else {
        None
    };
    if let Some(u) = &upscaler {
        let (render_w, render_h) = u.render_dims();
        tracing::info!(
            "DirectX: temporal upscaling active: scene render {}x{}, drawable {}x{}",
            render_w,
            render_h,
            width,
            height
        );
    }
    Ok(UpscaleState {
        backend: upscaler,
        requested: post.upscale_backend,
        jitter: std::cell::Cell::new([0.0, 0.0]),
        prev_elapsed: std::cell::Cell::new(0.0),
    })
}

// The fixed descriptor slots of the live-toggleable Quality effects, stashed so
// the runtime `apply_quality_settings` can build a launched-off feature into
// its slot without re-deriving the heap layout. The init builds take their
// slots from here too.
pub(super) fn build_quality_slots(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    swapchain: &SwapchainState,
    targets: &DxTargets,
    rtv: &RtvHeapLayout,
) -> QualitySlotHandles {
    let layout = &descriptors.layout;
    let srv = |slot| (descriptors.slot_cpu(slot), descriptors.slot_gpu(slot));
    let dsv_descriptor_size =
        heaps::descriptor_size(&gpu.hw.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    QualitySlotHandles {
        ssao_ao_raw_rtv: swapchain.rtv(rtv.ssao_base_slot),
        ssao_ao_raw_srv: srv(layout.ssao_srv_base_slot),
        ssao_ao_rtv: swapchain.rtv(rtv.ssao_base_slot + 1),
        ssao_ao_srv: srv(layout.ssao_srv_base_slot + 1),
        rt_output_rtv: swapchain.rtv(rtv.rt_output_slot),
        rt_output_srv: srv(layout.rt_output_srv_slot),
        refl_composite: ReflectionCompositeSlots {
            output_rtv: swapchain.rtv(rtv.refl_composite_base_slot),
            output_srv: srv(layout.refl_composite_srv_base_slot),
            blur_rtv: swapchain.rtv(rtv.refl_composite_base_slot + 1),
            blur_srv: srv(layout.refl_composite_srv_base_slot + 1),
        },
        gbuffer: GbufferSlots {
            normal_depth_rtv: swapchain.rtv(rtv.gbuffer_base_slot),
            normal_depth_srv: srv(layout.gbuffer_srv_base_slot),
            roughness_rtv: swapchain.rtv(rtv.gbuffer_base_slot + 1),
            roughness_srv: srv(layout.gbuffer_srv_base_slot + 1),
            velocity_rtv: swapchain.rtv(rtv.gbuffer_base_slot + 2),
            velocity_srv: srv(layout.gbuffer_srv_base_slot + 2),
            depth_dsv: heaps::cpu_handle(
                &targets.depth.heap,
                dsv_descriptor_size,
                DSV_GBUFFER_DEPTH_SLOT,
            ),
        },
    }
}

// The device every shared post pass builds its pipelines and targets through
// at init. Nothing encodes before the context exists, so it carries no probe
// set.
pub(super) fn post_device<'a>(
    gpu: &InitGpu<'a>,
    descriptors: &'a DxDescriptors,
    post: &'a PostDescriptors,
) -> DxPostDevice<'a> {
    DxPostDevice {
        device: &gpu.hw.device,
        descriptors: post,
        srv_heap: &descriptors.srv_heap,
        info_queue: gpu.hw.info_queue.as_ref(),
        probes: None,
        hot_reload: gpu.hot_reload,
    }
}

// TAA history-resolve resources. Sized at render-res; the motion it
// reprojects through comes from the unified G-buffer pre-pass.
pub(super) fn build_taa(
    post_device: &DxPostDevice<'_>,
    features: &Features,
    targets: &DxTargets,
) -> RenderResult<Option<TaaResources>> {
    let taa = if features.taa_enabled {
        Some(TaaResources::new(
            post_device,
            targets.extent.render_width,
            targets.extent.render_height,
        )?)
    } else {
        None
    };
    Ok(taa)
}

// SSAO: 1x1 white fallback always populated so the main pass binds a
// pass-through occlusion when SSAO is off. The real SSAO targets sit in
// the slots before it.
pub(super) fn build_ssao(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    slots: &QualitySlotHandles,
    settings: Option<SsaoSettings>,
) -> RenderResult<SsaoState> {
    let hw = gpu.hw;
    let device = hw.alloc.device();
    let white_slot = descriptors.layout.ssao_white_srv_slot;
    let ssao_white = create_fallback_white_resource(&hw.alloc)?;
    write_texture_srv(device, &ssao_white, descriptors.slot_cpu(white_slot));
    let ssao = if let Some(settings) = settings {
        let ao_resource = targets
            .transient_pool
            .resource_for("ao_output")
            .ok_or_else(|| {
                RenderError::Other("transient pool missing ao_output while SSAO is enabled".into())
            })?;
        Some(SsaoResources::new(
            SsaoDeviceCtx {
                device,
                info_queue: hw.info_queue.as_ref(),
            },
            targets.extent.render_width,
            targets.extent.render_height,
            settings,
            SsaoDescriptorHandles {
                ao_raw_rtv: slots.ssao_ao_raw_rtv,
                ao_raw_srv: slots.ssao_ao_raw_srv,
                ao_rtv: slots.ssao_ao_rtv,
                ao_srv: slots.ssao_ao_srv,
            },
            ao_resource,
            gpu.hot_reload,
        )?)
    } else {
        None
    };
    Ok(SsaoState {
        resources: ssao,
        white: ssao_white,
        white_srv_gpu: descriptors.slot_gpu(white_slot),
    })
}

// SSR: a fullscreen resolve reading the unified G-buffer pre-pass. The
// resources are built whenever SSR *or* SSGI *or* RT reflections are on (all
// reuse the G-buffer); `post.ssr` (the resolve half) stays `None` for a
// SSGI-only or RT-only build, which then holds no resolve or target.
pub(super) fn build_ssr(
    post_device: &DxPostDevice<'_>,
    targets: &DxTargets,
    post: &PostSettings,
) -> RenderResult<Option<SsrResources>> {
    let ssr = if post.ssr.is_some() || post.ssgi.is_some() || post.rt_reflections.is_some() {
        Some(SsrResources::new(
            post_device,
            targets.extent.render_width,
            targets.extent.render_height,
            post.ssr,
        )?)
    } else {
        None
    };
    Ok(ssr)
}

// SSGI: hemisphere-gather + depth-aware blur over the unified G-buffer
// pre-pass. The gather target lives in the shared pass; the composite blends
// straight into the scene.
pub(super) fn build_ssgi(
    post_device: &DxPostDevice<'_>,
    targets: &DxTargets,
    post: &PostSettings,
) -> RenderResult<Option<SsgiResources>> {
    let ssgi = match post.ssgi {
        Some(settings) => Some(SsgiResources::new(
            post_device,
            targets.extent.render_width,
            targets.extent.render_height,
            settings,
        )?),
        None => None,
    };
    Ok(ssgi)
}

// Unified G-buffer pre-pass resources. Built whenever any screen-space
// consumer drives it (see `Features::gbuffer_enabled`). Its three MRT RTVs sit
// at the tail of the RTV heap (after the decal RTV), its private depth DSV
// right after the shadow DSVs, and its three SRVs in the reserved
// `gbuffer_srv_base_slot` block. The skinned PSO builds lazily in
// `upload_skinned` once the joint-bound vertex layout exists.
pub(super) fn build_gbuffer(
    gpu: &InitGpu<'_>,
    targets: &DxTargets,
    slots: &QualitySlotHandles,
    gbuffer_enabled: bool,
) -> RenderResult<Option<GbufferResources>> {
    let gbuffer = if gbuffer_enabled {
        // The three color targets are pooled, so the pool (built with the
        // render targets, before this) is what owns them.
        let pooled = targets.transient_pool.gbuffer_pooled().ok_or_else(|| {
            RenderError::Other("transient pool missing the gbuffer color targets".into())
        })?;
        Some(GbufferResources::new(
            crate::directx::post::gbuffer::GbufferDeviceCtx {
                alloc: &gpu.hw.alloc,
            },
            crate::directx::post::gbuffer::GbufferExtent {
                width: targets.extent.render_width,
                height: targets.extent.render_height,
            },
            slots.gbuffer,
            &pooled,
        )?)
    } else {
        None
    };
    Ok(gbuffer)
}

// Projected decals: pipeline + unit-cube buffers + per-frame uniform rings.
// Always built so runtime `add_decal` works from a world that started with
// none; pre-authored decals get their albedo SRV written here.
pub(super) fn build_decals(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    scene: &DxSceneAssets,
    decals: Vec<DecalRecord>,
) -> RenderResult<DecalState> {
    let hw = gpu.hw;
    let decal_srv_base_slot = descriptors.layout.decal_srv_base_slot;
    let decals_state = Some(crate::directx::decal::DecalResources::new(
        &hw.alloc,
        targets.hdr.msaa_samples,
        decal_srv_base_slot,
        targets.main_depth_srv_gpu,
        hw.info_queue.as_ref(),
        gpu.hot_reload,
    )?);
    // Pre-authored decals: write each one's albedo SRV into its reserved
    // heap slot. Runtime adds via `DxContext::add_decal` follow the same
    // pattern.
    if decals.len() > crate::directx::decal::MAX_DECALS {
        return Err(RenderError::Other(format!(
            "decals: {} authored decals exceed MAX_DECALS ({})",
            decals.len(),
            crate::directx::decal::MAX_DECALS
        )));
    }
    let last_tex = scene.textures.len().saturating_sub(1);
    for (i, rec) in decals.iter().enumerate() {
        let tex_idx = rec.texture_slot.min(last_tex);
        write_texture_srv(
            &hw.device,
            &scene.textures[tex_idx],
            descriptors.slot_cpu(decal_srv_base_slot + i),
        );
    }
    // The slot table the decal pass draws from. Each authored decal takes
    // the slot whose albedo SRV was just written above, in the same order.
    let mut decal_set = decal::DecalSet::new(crate::directx::decal::MAX_DECALS, FRAMES);
    for record in decals {
        decal_set.insert(record).map_err(|_| {
            RenderError::Other("decals: authored decals exceed MAX_DECALS".to_string())
        })?;
    }
    Ok(DecalState {
        state: decals_state,
        set: decal_set,
    })
}

// Volumetric fog: pipeline + per-frame uniform ring. Built only when the world
// declared a `VolumetricFog`; the encoder simply skips the pass when
// `settings` is `None`. The fog pass shares the main-depth SRV the decal pass
// binds.
pub(super) fn build_fog(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    shadow: &ShadowState,
    settings: Option<FogSettings>,
    light_uniforms: &LightUniforms,
) -> RenderResult<FogState> {
    let hw = gpu.hw;
    let layout = &descriptors.layout;
    let fog_resources = if settings.is_some() {
        Some(crate::directx::fog::FogResources::new(
            &hw.alloc,
            crate::directx::fog::FogVolumeDescriptors {
                uav_cpu: descriptors.slot_cpu(layout.fog_froxel_uav_slot),
                uav_gpu: descriptors.slot_gpu(layout.fog_froxel_uav_slot),
                srv_cpu: descriptors.slot_cpu(layout.fog_froxel_srv_slot),
                srv_gpu: descriptors.slot_gpu(layout.fog_froxel_srv_slot),
            },
            crate::directx::fog::FogShaderResourceHandles {
                depth_srv_gpu: targets.main_depth_srv_gpu,
                shadow_srv_gpu: shadow.srv_gpu,
            },
            crate::directx::fog::FogDeviceParams {
                msaa_samples: targets.hdr.msaa_samples,
                hot_reload: gpu.hot_reload,
            },
            hw.info_queue.as_ref(),
        )?)
    } else {
        None
    };
    // The first directional light's direction and color * intensity, cached
    // for the volumetric-fog encoder since `LightUniforms` is uploaded rather
    // than pushed each frame. `update_directional_lights` re-derives both.
    Ok(FogState {
        resources: fog_resources,
        settings,
        sun_dir: shadow.light_dir,
        sun_color: lights::sun_color(light_uniforms),
    })
}

// Particles: compute + render pipelines + per-frame uniform rings, plus one
// persistent GPU pool per emitter. Built only when the world declared at least
// one emitter; the encoder skips the passes when `resources` is `None`, and
// runtime `add_emitter` builds the pipelines lazily the same way. The emitter
// cap matches the SRV-heap reservation.
pub(super) fn build_particles(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    scene: &DxSceneAssets,
    particles: Vec<ParticleEmitterRecord>,
) -> RenderResult<ParticleState> {
    let hw = gpu.hw;
    let particle_srv_base_slot = descriptors.layout.particle_srv_base_slot;
    if particles.len() > crate::directx::particle::MAX_EMITTERS {
        return Err(RenderError::Other(format!(
            "particles: {} authored emitters exceed MAX_EMITTERS ({})",
            particles.len(),
            crate::directx::particle::MAX_EMITTERS
        )));
    }
    let (particle_resources, particle_records, particle_emitter_states) = if !particles.is_empty() {
        let resources = crate::directx::particle::ParticleResources::new(
            &hw.alloc,
            particle_srv_base_slot,
            hw.info_queue.as_ref(),
            gpu.hot_reload,
        )?;
        let mut states: Vec<Option<crate::directx::particle::ParticleEmitterGpuState>> =
            Vec::with_capacity(particles.len());
        let last_tex = scene.textures.len().saturating_sub(1);
        for (i, rec) in particles.iter().enumerate() {
            let state = crate::directx::particle::build_emitter_gpu_state(&hw.alloc, rec)?;
            states.push(Some(state));
            // Write the per-emitter albedo SRV into its reserved heap slot.
            let tex_idx = rec.texture_slot.min(last_tex);
            write_texture_srv(
                &hw.device,
                &scene.textures[tex_idx],
                descriptors.slot_cpu(particle_srv_base_slot + i),
            );
        }
        let recs: Vec<Option<particles::ParticleEmitterRecord>> =
            particles.into_iter().map(Some).collect();
        (Some(resources), recs, states)
    } else {
        (None, Vec::new(), Vec::new())
    };
    Ok(ParticleState {
        resources: particle_resources,
        records: particle_records,
        emitter_state: particle_emitter_states,
        free_slots: Vec::new(),
        srv_base_slot: particle_srv_base_slot,
        last_elapsed: std::cell::Cell::new(0.0),
        frame_index: std::cell::Cell::new(0),
    })
}

// Auto-exposure: build the histogram + average compute pipelines plus
// the GPU buffers (histogram UAV, output UAV, per-frame readback)
// only when the world's PostProcessConfig opted in. With auto-exposure
// off every path below is None and the static authored EV continues
// to drive `post_process.exposure` unchanged.
pub(super) fn build_auto_exposure(
    gpu: &InitGpu<'_>,
    post: &PostSettings,
) -> RenderResult<AutoExposureState> {
    let hw = gpu.hw;
    let (resources, state) = if let Some(settings) = post.auto_exposure.as_ref() {
        let resources = dump_on_err(
            hw.info_queue.as_ref(),
            crate::directx::auto_exposure::AutoExposureResources::new(&hw.alloc, gpu.hot_reload),
        )?;
        let state = auto_exposure::AutoExposureState::new(settings);
        (Some(resources), Some(state))
    } else {
        (None, None)
    };
    Ok(AutoExposureState {
        resources,
        settings: post.auto_exposure,
        state,
        bias_ev: post.auto_exposure_bias_ev,
        last_elapsed: 0.0,
    })
}

// Raymarched SDF volumes. Builds per-volume PSOs from `.hlsl`
// payloads and writes the raymarch SRV + sampler tables into
// their reserved blocks. `.metal` payloads are filtered out
// inside `try_new` with a logged warning; if every volume is
// Metal-first (the current showcase shape), this returns `None`
// and the render graph never adds `PassId::Raymarch`. The
// shadow + IBL handles passed here mirror the matching slot-0/1/2
// bindings the main pass uses, so raymarched surfaces sample the
// same CSM cascades + IBL cubes as rasterized geometry.
pub(super) fn build_raymarch(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    shadow: &ShadowState,
    scene: &DxSceneAssets,
    sdf_volumes: &[(SdfVolume, Vec<u8>, String)],
) -> RenderResult<Option<RaymarchResources>> {
    let hw = gpu.hw;
    let raymarch_srv_base_slot = descriptors.layout.raymarch_srv_base_slot;
    let sampler_descriptor_size =
        heaps::descriptor_size(&hw.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    let raymarch = RaymarchResources::try_new(
        crate::directx::raymarch::RaymarchDeviceContext {
            alloc: &hw.alloc,
            info_queue: hw.info_queue.as_ref(),
        },
        crate::directx::raymarch::RaymarchTargetConfig {
            width: targets.extent.render_width,
            height: targets.extent.render_height,
            msaa_samples: targets.hdr.msaa_samples,
        },
        crate::directx::raymarch::RaymarchSharedBindings {
            shadow_resource: shadow.resource.as_ref().map(|r| &r.resource),
            shadow_layers: NUM_SHADOW_CASCADES as u32,
            irradiance_resource: &scene.env_map.irradiance.resource,
            prefilter_resource: &scene.env_map.prefilter.resource,
        },
        crate::directx::raymarch::RaymarchDescriptorHandles {
            srv_base_cpu: descriptors.slot_cpu(raymarch_srv_base_slot),
            srv_base_gpu: descriptors.slot_gpu(raymarch_srv_base_slot),
            srv_descriptor_size: descriptors.srv_descriptor_size,
            sampler_base_cpu: heaps::cpu_handle(
                &descriptors.sampler_heap,
                sampler_descriptor_size,
                heaps::RAYMARCH_SAMPLER_BASE_SLOT,
            ),
            sampler_base_gpu: heaps::gpu_handle(
                &descriptors.sampler_heap,
                sampler_descriptor_size,
                heaps::RAYMARCH_SAMPLER_BASE_SLOT,
            ),
            sampler_descriptor_size,
        },
        sdf_volumes,
        gpu.hot_reload,
    )?;
    Ok(raymarch)
}

// Planar reflections: group each transparent reflector's plane into a
// bounded set of distinct planes (near-coplanar reflectors share one mirror
// render; reflectors past the budget fall back to the probe cube). The
// distinct count sizes the reserved planar-resolve SRV block; `slots[i]` is
// reflector `i`'s resolve slot (or `None`). Planned before the descriptor
// heap is created so the block is sized first.
//
// Water first, then glass, matching the Metal backend, so the two slot
// ranges are `[..water_surfaces.len()]` and the rest.
pub(super) fn plan_planar(fx: &WorldFx, planar_planes: usize) -> PlanarAssignment {
    let planar_panes: Vec<[f32; 4]> = fx
        .water_surfaces
        .iter()
        // A water surface's rest plane: horizontal at the surface base height.
        .map(|s| [0.0, 1.0, 0.0, -s.center[1]])
        .chain(
            fx.glass_panels
                .iter()
                .map(|p| crate::directx::planar::pane_plane(p.normal, p.center)),
        )
        .collect();
    // Cap at the capacity ceiling the reserved planar resolve SRVs are sized to,
    // so a stale/over-large preset value can never over-allocate.
    let planar_budget = planar_planes.min(crate::directx::planar::MAX_PLANAR_PLANES);
    planar_reflection::assign_planar_slots(&planar_panes, planar_budget)
}

// One mirror-render resolve per distinct reflector plane (the
// `assign_planar_slots` representatives), each SRV in a reserved heap slot the
// transparent pass binds per record. `None` when no reflector was assigned a
// planar slot (no transparent content, or every plane degenerate / over
// budget). `n_cull` is the build-time draw-record count (matches
// `DxContext::cull_count`), which sizes each plane's region of the mirror-cull
// indirect buffer.
pub(super) fn build_planar_reflection(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    planar: &PlanarAssignment,
    n_cull: usize,
    clear_color: [f32; 4],
) -> RenderResult<Option<PlanarReflectionSet>> {
    let planar_resolve_srv_base_slot = descriptors.layout.planar_resolve_srv_base_slot;
    let planar_reflection = if planar.representatives.is_empty() {
        None
    } else {
        let resolve_srv_cpu: Vec<_> = (0..planar.representatives.len())
            .map(|i| descriptors.slot_cpu(planar_resolve_srv_base_slot + i))
            .collect();
        let resolve_srv_gpu: Vec<_> = (0..planar.representatives.len())
            .map(|i| descriptors.slot_gpu(planar_resolve_srv_base_slot + i))
            .collect();
        Some(PlanarReflectionSet::new(
            &gpu.hw.alloc,
            crate::directx::planar::PlanarConfig {
                sample_count: targets.hdr.msaa_samples,
                width: targets.extent.render_width,
                height: targets.extent.render_height,
                n_cull,
            },
            &planar.representatives,
            crate::directx::planar::PlanarTargets {
                resolve_srv_cpu: &resolve_srv_cpu,
                resolve_srv_gpu: &resolve_srv_gpu,
                clear_color,
            },
        )?)
    };
    Ok(planar_reflection)
}

pub(super) struct TransparentInputs<'a> {
    pub(super) descriptors: &'a DxDescriptors,
    pub(super) targets: &'a DxTargets,
    pub(super) planar: &'a PlanarAssignment,
    pub(super) glass_panels: &'a [GlassPanel],
    pub(super) water_surfaces: &'a [WaterSurface],
    pub(super) draw_objects: &'a [DrawObject],
}

// The shared transparent pass and its producers: water surfaces, translucent
// glass panes, and see-through glass meshes. `Some` only when the world
// declared at least one of the three; the mesh case additionally needs a
// DXR-capable GPU, since its producer is ray-traced only and a pane-less,
// water-less world would otherwise build the whole pass for a producer that
// cannot exist. Shares the main-depth SRV with the decal pass; the scene-copy
// snapshot uses its own reserved heap slot. `planar.slots` gives each
// reflector its planar resolve slot (or `None` -> probe-cube fallback),
// numbered water first to match `plan_planar`.
pub(super) fn build_transparent(
    gpu: &InitGpu<'_>,
    inputs: TransparentInputs<'_>,
) -> RenderResult<Option<TransparentResources>> {
    let TransparentInputs {
        descriptors,
        targets,
        planar,
        glass_panels,
        water_surfaces,
        draw_objects,
    } = inputs;
    let hw = gpu.hw;
    let scene_copy_slot = descriptors.layout.transparent_scene_copy_srv_slot;
    // Layer 2 see-through glass is opt-in per `Material` (the `see_through`
    // arg, which implies `transparent`): see-through only looks right when the
    // space behind the glass is modeled. A material that is `transparent` but
    // NOT `see_through` renders as Layer 1 (opaque, low roughness, scene
    // reflections) = tinted reflective glass that hides the interior. This list
    // drives the transparent-pass producer, the opaque-pass skip and the
    // RT-BLAS exclude together.
    let seethrough_mesh_indices: Vec<usize> = draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
        .map(|(i, _)| i)
        .collect();
    let has_seethrough_meshes = !seethrough_mesh_indices.is_empty() && hw.rt_capable;
    let transparent =
        if glass_panels.is_empty() && water_surfaces.is_empty() && !has_seethrough_meshes {
            None
        } else {
            let (water_planar_slots, glass_planar_slots) =
                planar.slots.split_at(water_surfaces.len());
            Some(TransparentResources::new(
                crate::directx::transparent::TransparentDeviceCtx { alloc: &hw.alloc },
                crate::directx::transparent::TransparentBuildConfig {
                    msaa_samples: targets.hdr.msaa_samples,
                    width: targets.extent.render_width,
                    height: targets.extent.render_height,
                    hot_reload: gpu.hot_reload,
                },
                crate::directx::transparent::TransparentSceneTargets {
                    scene_copy_srv_cpu: descriptors.slot_cpu(scene_copy_slot),
                    scene_copy_srv_gpu: descriptors.slot_gpu(scene_copy_slot),
                    depth_srv_gpu: targets.main_depth_srv_gpu,
                },
                crate::directx::transparent::TransparentContent {
                    glass_panels,
                    glass_planar_slots,
                    water_surfaces,
                    water_planar_slots,
                    seethrough_mesh_indices: &seethrough_mesh_indices,
                },
                hw.info_queue.as_ref(),
            )?)
        };
    Ok(transparent)
}

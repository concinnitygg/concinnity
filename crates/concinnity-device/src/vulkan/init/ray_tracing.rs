//! Ray-traced reflections: the scene acceleration structure, the reflection pass
//! that traces it, and the reflection composite the SSR resolve shares.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::init::{RenderLaunch, SceneData};
use concinnity_engine::render::buffers::retire_pool::RetirePool;
use concinnity_engine::render::config::ResolvedGraphics;

use super::InitGpu;
use super::effects::shared_post_device;
use crate::vulkan::context::{CullState, Descriptors, Geometry, RtState, SceneAssets, Targets};
use crate::vulkan::post::PostSupport;
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::post::reflection_composite::{
    VkReflectionCompositePass, build_reflection_composite,
};
use crate::vulkan::post::rt_reflections::RtReflectionsResources;
use concinnity_engine::render::post::reflection_path::ReflectionPath;

#[derive(Clone, Copy)]
pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) geometry: &'a Geometry,
    pub(super) scene: &'a SceneAssets,
    pub(super) targets: &'a Targets,
    pub(super) gbuffer: Option<&'a GbufferResources>,
    pub(super) descriptors: &'a Descriptors,
    pub(super) cull: &'a CullState,
    pub(super) graphics: &'a ResolvedGraphics,
    pub(super) launch: RenderLaunch,
    pub(super) post_support: &'a PostSupport,
    pub(super) rt_wanted: bool,
}

pub(super) struct RtResources {
    pub(super) state: RtState,
    pub(super) reflections: Option<RtReflectionsResources>,
    pub(super) composite: Option<VkReflectionCompositePass>,
    pub(super) seethrough_mesh_indices: Vec<usize>,
    pub(super) has_seethrough_meshes: bool,
}

pub(super) fn build_rt_reflections(
    gpu: &InitGpu<'_>,
    inputs: RtInputs<'_>,
) -> RenderResult<RtResources> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let RtInputs {
        world,
        geometry,
        scene,
        targets,
        gbuffer,
        descriptors,
        cull,
        graphics,
        launch,
        post_support,
        rt_wanted,
    } = inputs;
    let (hdr_resolve_images, render_extent) = (&targets.hdr_resolve_images, targets.render_extent);
    // Hardware ray-traced reflections and the acceleration structure, built when
    // the world asks and the device can ray query. A compile failure leaves the
    // pass `None` and the graph keeps `SsrResolve`. The BVH comes and goes with
    // resident geometry.
    //
    // See-through glass is opt-in per `Material::see_through`; this list drives the
    // transparent producer, the opaque-pass skip and the BLAS exclusion together.
    let seethrough_mesh_indices: Vec<usize> = world
        .draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
        .map(|(i, _)| i)
        .collect();

    // Whether those meshes reroute, decided before the BLAS build with the same
    // predicate as their producer. If the mesh shaders fail to compile they render
    // opaque but stay out of the BLAS until a topology refresh.
    let has_seethrough_meshes = !seethrough_mesh_indices.is_empty() && hw.rt_capable;

    let rt_opt = if rt_wanted {
        let hdr_views: Vec<vk::ImageView> = hdr_resolve_images.iter().map(|i| i.view).collect();
        // RT reads the unified G-buffer pre-pass's per-frame normal+depth +
        // roughness (built earlier whenever any consumer is on); `gbuffer` is
        // `Some` here because RT forces the pre-pass on.
        let gb = gbuffer
            .as_ref()
            .expect("RT forces the unified G-buffer pre-pass to exist");
        let nd_views = gb.normal_depth_views();
        let rough_views = gb.roughness_views();
        match RtReflectionsResources::new(
            crate::vulkan::post::rt_reflections::RtBuild {
                alloc,
                device,
                width: render_extent.width,
                height: render_extent.height,
                frames,
            },
            graphics
                .rt_reflections()
                .expect("rt_wanted implies rt_settings is Some"),
            crate::vulkan::post::rt_reflections::RtStaticInputs {
                vertex_buffer: geometry.vertex_buffer.buffer(),
                index_buffer: geometry.index_buffer.buffer(),
                hdr_resolve_views: &hdr_views,
                gbuffer_views: &nd_views,
                roughness_views: &rough_views,
            },
            crate::vulkan::post::rt_reflections::RtLayoutConfig {
                bindless_set_layout: cull
                    .bindless_set_layout
                    .as_ref()
                    .map(super::super::owned::OwnedSetLayout::handle),
                global_set_layout: descriptors.global_set_layout.handle(),
                pool_size: cull.bindless_pool_size,
                hot_reload,
            },
        ) {
            Ok(rt) => Some(rt),
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (falling back to SSR): {e}");
                None
            }
        }
    } else {
        None
    };
    let rt_accel_opt = if rt_opt.is_some() {
        match crate::vulkan::raytrace::build_rt_accel(
            crate::vulkan::raytrace::RtDeviceCtx {
                alloc,
                instance: &hw.instance,
                device,
                pd: hw.physical_device,
            },
            command_pool,
            hw.graphics_queue,
            crate::vulkan::raytrace::RtSceneGeometry {
                shared: crate::vulkan::raytrace::SharedGeometry::of(geometry),
                draw_objects: &world.draw_objects,
                clusters: &world.instanced_clusters,
                albedo_count: scene.textures.len(),
                exclude_seethrough: has_seethrough_meshes,
                // Skinned meshes upload after init; the first dynamic frame
                // adds them, or seeds a BVH for them when there is none.
                skinned_present: false,
            },
            frames,
        ) {
            Ok(Some(accel)) => Some(accel),
            Ok(None) => {
                tracing::info!(
                    "RT reflections requested but no resident triangle geometry to trace yet"
                );
                None
            }
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed: {e}");
                None
            }
        }
    } else {
        None
    };
    // Reflection composite: built whenever a resolve feeds it. Both resolves
    // write radiance+weight into their output target; this blurs by roughness
    // and composites over the scene into its own output, which then replaces
    // the raw resolve output as the scene image every downstream pass samples.
    let composite_opt = if ReflectionPath::new(
        graphics.ssr().is_some(),
        rt_opt.is_some(),
        rt_accel_opt.is_some(),
    )
    .composite()
    {
        Some(build_reflection_composite(
            &shared_post_device(gpu, post_support, scene, descriptors),
            graphics.reflection_blur_scale(),
            render_extent,
            rt_opt.as_ref().map(|rt| rt.settings.divisor),
        )?)
    } else {
        None
    };
    Ok(RtResources {
        state: RtState {
            accel: rt_accel_opt,
            retired: RetirePool::new(),
            retire_tick: 0,
            dynamic_mode: launch.rt_dynamic,
            skinned_geometry: launch.rt_skinned_geometry,
            update_streak: concinnity_engine::render::rt_accel::FailureStreak::default(),
            skin: rt_opt
                .is_some()
                .then(|| crate::vulkan::raytrace::build_rt_skin(alloc, device, hot_reload))
                .flatten(),
        },
        reflections: rt_opt,
        composite: composite_opt,
        seethrough_mesh_indices,
        has_seethrough_meshes,
    })
}

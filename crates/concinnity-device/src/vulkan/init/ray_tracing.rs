//! Ray-traced reflections: the scene acceleration structure, the reflection pass
//! that traces it, and the reflection composite the SSR resolve shares.

use ash::vk;
use concinnity_core::render::backend_init::{PostSettings, SceneData};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{
    VkCull, VkDescriptors, VkGeometry, VkRayTracing, VkSceneAssets, VkTargets,
};
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::post::rt_reflections::RtReflectionsResources;

pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) geometry: &'a VkGeometry,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) targets: &'a VkTargets,
    pub(super) gbuffer: Option<&'a GbufferResources>,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) cull: &'a VkCull,
    pub(super) post: &'a PostSettings,
    pub(super) rt_wanted: bool,
}

pub(super) struct RtResources {
    pub(super) state: VkRayTracing,
    pub(super) reflections: Option<RtReflectionsResources>,
    pub(super) composite: Option<ReflectionCompositeResources>,
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
        post,
        rt_wanted,
    } = inputs;
    let (hdr_resolve_images, render_extent) = (&targets.hdr_resolve_images, targets.render_extent);
    // Hardware ray-traced reflections: the scene acceleration structure +
    // the inline-`rayQueryEXT` reflection pass. Built only when the world
    // requested it AND the device exposed the ray-query extensions
    // (`rt_wanted`). Reuses the SSR pre-pass G-buffer (forced on earlier) for
    // the per-pixel surface point + normal, and the bindless pool (when live)
    // for textured hit shading. Graceful-fallback throughout: no resident
    // geometry, an AS build error, or a shader compile failure leaves both
    // `None` and the graph keeps `SsrResolve`. RT takes precedence over the
    // SSR resolve in the shared graph slot, which `ReflectionPath` settles once
    // the build outcome is known.
    // Layer 2 see-through glass is opt-in per `Material` (the `see_through`
    // arg, which implies `transparent`): see-through only looks right when the
    // space behind the glass is modeled. A material that is `transparent` but
    // NOT `see_through` renders as Layer 1 (opaque, low roughness, scene
    // reflections) = tinted reflective glass that hides the interior. This list
    // drives the transparent-pass producer, the opaque-pass skip and the
    // RT-BLAS exclude together.
    let seethrough_mesh_indices: Vec<usize> = world
        .draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
        .map(|(i, _)| i)
        .collect();

    // Whether those meshes will be rerouted, decided here because the initial
    // BLAS is built before the transparent pass that owns the mesh producer.
    // It is the same predicate that producer is gated on. The one divergence
    // is a mesh-shader compile failure, which leaves the producer absent (and
    // logs): the meshes then render opaque but stay out of this BLAS until a
    // topology refresh re-reads `seethrough_meshes_enabled` and puts them back.
    let has_seethrough_meshes = !seethrough_mesh_indices.is_empty() && hw.rt_capable;

    let (rt_accel_opt, rt_opt) = if rt_wanted {
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
                vertex_buffer: geometry.vertex_buffer.buffer(),
                index_buffer: geometry.index_buffer.buffer(),
                draw_objects: &world.draw_objects,
                clusters: &world.instanced_clusters,
                albedo_count: scene.textures.len(),
                total_vertices: world.vertices.len(),
                exclude_seethrough: has_seethrough_meshes,
            },
            frames,
            hot_reload,
        ) {
            Ok(Some(accel)) => {
                let hdr_views: Vec<vk::ImageView> =
                    hdr_resolve_images.iter().map(|i| i.view).collect();
                // RT reads the unified G-buffer pre-pass's per-frame
                // normal+depth + roughness (built earlier whenever any consumer
                // is on); `gbuffer` is `Some` here because RT forces the
                // pre-pass on.
                let gb = gbuffer
                    .as_ref()
                    .expect("RT forces the unified G-buffer pre-pass to exist");
                let nd_views = gb.normal_depth_views();
                let rough_views = gb.roughness_views();
                let (geom_buffer, geom_size) = accel.geom_table();
                match crate::vulkan::post::rt_reflections::RtReflectionsResources::new(
                    crate::vulkan::post::rt_reflections::RtBuild {
                        alloc,
                        device,
                        width: render_extent.width,
                        height: render_extent.height,
                        frames,
                    },
                    post.rt_reflections
                        .expect("rt_wanted implies rt_settings is Some"),
                    crate::vulkan::post::rt_reflections::RtStaticInputs {
                        vertex_buffer: geometry.vertex_buffer.buffer(),
                        index_buffer: geometry.index_buffer.buffer(),
                        hdr_resolve_views: &hdr_views,
                        gbuffer_views: &nd_views,
                        roughness_views: &rough_views,
                    },
                    crate::vulkan::post::rt_reflections::RtAccelHandles {
                        tlas: accel.tlas(),
                        geom_buffer,
                        geom_size,
                        deformed_verts: accel.deformed_verts(),
                        skinned_indices: accel.skinned_indices(),
                    },
                    crate::vulkan::post::rt_reflections::RtLayoutConfig {
                        bindless_set_layout: cull.bindless_set_layout.as_ref().map(|l| l.handle()),
                        global_set_layout: descriptors.global_set_layout.handle(),
                        pool_size: cull.bindless_pool_size,
                        hot_reload,
                    },
                ) {
                    Ok(rt) => (Some(accel), Some(rt)),
                    Err(e) => {
                        tracing::warn!(
                            "RT reflections pass build failed (falling back to SSR): {e}"
                        );
                        let mut accel = accel;
                        accel.destroy(device);
                        (None, None)
                    }
                }
            }
            Ok(None) => {
                tracing::info!(
                    "RT reflections requested but no resident triangle geometry to trace; \
                     using SSR"
                );
                (None, None)
            }
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed (falling back to SSR): {e}");
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let rt_active = rt_opt.is_some();
    // Reflection composite: built whenever a resolve feeds it. Both resolves
    // write radiance+weight into their output target; this blurs by roughness
    // and composites over the scene into its own output, which then replaces
    // the raw resolve output as the scene image every downstream pass samples.
    let composite_opt = if crate::vulkan::post::reflection_composite::ReflectionPath::new(
        post.ssr.is_some(),
        rt_active,
    )
    .composite
    {
        let gb = gbuffer
            .as_ref()
            .expect("a reflection path implies the unified G-buffer pre-pass");
        Some(
            crate::vulkan::post::reflection_composite::ReflectionCompositeResources::new(
                &gpu.upload(),
                render_extent.width,
                render_extent.height,
                frames,
                post.reflection_blur_scale,
                &crate::vulkan::post::reflection_composite::CompositeInputs::new(
                    hdr_resolve_images,
                    gb,
                ),
                hot_reload,
            )?,
        )
    } else {
        None
    };
    Ok(RtResources {
        state: VkRayTracing {
            accel: rt_accel_opt,
            dynamic_mode: post.rt_dynamic,
            skinned_geometry: post.rt_skinned_geometry,
            topology_dirty: false,
            static_vertex_count: world.vertices.len(),
        },
        reflections: rt_opt,
        composite: composite_opt,
        seethrough_mesh_indices,
        has_seethrough_meshes,
    })
}

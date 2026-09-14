//! Ray tracing: the roughness-aware reflection composite the SSR and RT
//! resolves feed, the RT reflection pipelines with their output target, and the
//! acceleration structure they trace.

use concinnity_core::render::backend_init::{PostSettings, SceneData};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::directx::context::{DxRayTracing, DxSceneAssets, DxTargets};
use crate::directx::post::reflection_composite::ReflectionCompositeResources;
use crate::directx::post::rt_reflections::{
    RtBuildContext, RtBuildInit, RtOutputDescriptors, RtReflectionsResources,
};
use crate::directx::quality::QualitySlotHandles;
use crate::directx::raytrace;
use crate::directx::transparent::TransparentResources;

// Reflection composite: `output` is the scene-with-reflections the post stack
// consumes; `blur` is the reduced-res roughness blur. Built when SSR resolve
// or RT is authored (both feed the same composite); the slots stay reserved
// either way for a live reflection enable.
pub(super) fn build_reflection_composite(
    gpu: &InitGpu<'_>,
    slots: &QualitySlotHandles,
    targets: &DxTargets,
    post: &PostSettings,
) -> RenderResult<Option<ReflectionCompositeResources>> {
    let hw = gpu.hw;
    let reflection_composite = if post.ssr.is_some() || post.rt_reflections.is_some() {
        Some(ReflectionCompositeResources::new(
            &hw.device,
            targets.extent.render_width,
            targets.extent.render_height,
            post.reflection_blur_scale,
            slots.refl_composite,
            hw.info_queue.as_ref(),
            gpu.hot_reload,
        )?)
    } else {
        None
    };
    Ok(reflection_composite)
}

// RT reflections: build the pipelines + output target only when authored AND
// the GPU supports the DXR tier. The DXC compile can still fail (DXC DLL
// absent / shader error); that is non-fatal -> leave `None` and the graph
// falls back to SSR (whose resolve is also off in an RT-only world, so the
// scene simply renders without reflections). The acceleration structure is
// built separately and gates `rt_reflections_active` alongside this.
pub(super) fn build_rt_reflections(
    gpu: &InitGpu<'_>,
    slots: &QualitySlotHandles,
    targets: &DxTargets,
    post: &PostSettings,
) -> Option<RtReflectionsResources> {
    let hw = gpu.hw;
    match (post.rt_reflections, hw.rt_capable) {
        (Some(settings), true) => match RtReflectionsResources::new(
            RtBuildContext {
                alloc: &hw.alloc,
                width: targets.extent.render_width,
                height: targets.extent.render_height,
            },
            settings,
            RtOutputDescriptors {
                output_rtv: slots.rt_output_rtv,
                output_srv: slots.rt_output_srv,
            },
            RtBuildInit {
                info_queue: hw.info_queue.as_ref(),
                hot_reload: gpu.hot_reload,
            },
        ) {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!("RT reflections unavailable, falling back to SSR: {e}");
                None
            }
        },
        _ => None,
    }
}

pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) scene: &'a DxSceneAssets,
    pub(super) reflections: Option<&'a RtReflectionsResources>,
    pub(super) transparent: Option<&'a TransparentResources>,
    pub(super) post: &'a PostSettings,
}

// Hardware-RT acceleration structure. Built once over the shared static
// vertex/index buffers + the draw-object / cluster lists, only when the
// RT reflection resources came up (DXR-capable GPU + DXC compile OK).
// `Ok(None)` means an empty scene; an `Err` is non-fatal (logged, falls
// back to SSR). `rt_reflections_active` gates the RT pass on both this
// and the resources being `Some`. The init build is static-only; skinned
// meshes are seeded into the BVH on the first dynamic frame
// (`rebuild_skinned`), so the compute-skinning pipeline is built here and
// attached. A skin-pipeline build failure is non-fatal: the
// RT pass still runs for static geometry, just without skinned hits.
pub(super) fn build_ray_tracing(gpu: &InitGpu<'_>, inputs: RtInputs<'_>) -> DxRayTracing {
    let RtInputs {
        world,
        scene,
        reflections,
        transparent,
        post,
    } = inputs;
    let hw = gpu.hw;
    let accel = if reflections.is_some() {
        match raytrace::build_rt_accel(raytrace::RtInitGeometry {
            alloc: &hw.alloc,
            vertex_buffer: &scene.geometry.vertex_buffer,
            index_buffer: &scene.geometry.index_buffer,
            draw_objects: &world.draw_objects,
            clusters: &world.instanced_clusters,
            total_vertices: world.vertices.len(),
            albedo_count: scene.textures.len() as u32,
            // Exclude the meshes the transparent pass will reroute. Decided
            // here rather than through `seethrough_meshes_enabled` because
            // the context does not exist yet; the two agree because both read
            // "a material opted in AND the mesh pipelines built".
            exclude_seethrough: transparent.is_some_and(|t| t.mesh_pipelines_ready()),
        }) {
            Ok(Some(mut accel)) => {
                match raytrace::build_rt_skin_pipeline(&hw.device, gpu.hot_reload) {
                    Ok(skin) => accel.set_skin_pipeline(skin),
                    Err(e) => tracing::warn!(
                        "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
                    ),
                }
                Some(accel)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed, falling back to SSR: {e}");
                None
            }
        }
    } else {
        None
    };
    DxRayTracing {
        accel,
        dynamic_mode: post.rt_dynamic,
        skinned_geometry: post.rt_skinned_geometry,
        topology_dirty: false,
        static_vertex_count: world.vertices.len(),
    }
}

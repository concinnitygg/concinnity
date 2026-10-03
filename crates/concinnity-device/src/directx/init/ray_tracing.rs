//! Ray tracing: the roughness-aware reflection composite the SSR and RT
//! resolves feed, the RT reflection pipelines with their output target, and the
//! acceleration structure they trace.

use concinnity_core::render::backend_init::{PostSettings, SceneData};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::directx::context::{DxRayTracing, DxSceneAssets, DxTargets};
use crate::directx::post::post_device::DxPostDevice;
use crate::directx::post::reflection_composite::{
    DxReflectionCompositePass, build_reflection_composite as build_composite,
};
use crate::directx::post::rt_reflections::{
    RtBuildContext, RtBuildInit, RtOutputDescriptors, RtReflectionsResources,
};
use crate::directx::quality::QualitySlotHandles;
use crate::directx::raytrace;
use crate::directx::transparent::TransparentResources;

// Reflection composite: its output is the scene-with-reflections the post stack
// consumes. Built when SSR resolve or RT is authored (both feed the same
// composite); its targets come from the post block, so a live reflection
// enable builds it the same way.
pub(super) fn build_reflection_composite(
    device: &DxPostDevice,
    targets: &DxTargets,
    post: &PostSettings,
) -> RenderResult<Option<DxReflectionCompositePass>> {
    if post.ssr.is_none() && post.rt_reflections.is_none() {
        return Ok(None);
    }
    Ok(Some(build_composite(
        device,
        post.reflection_blur_scale,
        targets.extent.render_width,
        targets.extent.render_height,
    )?))
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
// and the resources being `Some`. Skinned meshes upload after init, so this
// build covers static geometry; the first dynamic frame adds the skinned
// BLAS (`rebuild_skinned`), or seeds a BVH for them when there is none.
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
            shared: raytrace::SharedGeometry::of(&scene.geometry),
            draw_objects: &world.draw_objects,
            clusters: &world.instanced_clusters,
            albedo_count: scene.textures.len() as u32,
            // Exclude the meshes the transparent pass will reroute. Decided
            // here rather than through `seethrough_meshes_enabled` because
            // the context does not exist yet; the two agree because both read
            // "a material opted in AND the mesh pipelines built".
            exclude_seethrough: transparent.is_some_and(|t| t.mesh_pipelines_ready()),
            skinned_present: false,
        }) {
            Ok(accel) => accel,
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed, falling back to SSR: {e}");
                None
            }
        }
    } else {
        None
    };
    let skin = reflections
        .is_some()
        .then(|| raytrace::build_rt_skin(&hw.device, gpu.hot_reload))
        .flatten();
    DxRayTracing {
        accel,
        dynamic_mode: post.rt_dynamic,
        update_streak: Default::default(),
        retired: Default::default(),
        retire_tick: 0,
        skinned_geometry: post.rt_skinned_geometry,
        skin,
    }
}

//! Hardware ray-traced reflections: the resolve and skinning pipelines, and the
//! scene acceleration structure built over the uploaded geometry.

use concinnity_core::render::backend_init::{PostSettings, SceneData};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::rt_reflections::RtReflectionSettings;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::InitGpu;
use crate::metal::context::{GlassState, MtlSceneAssets};
use crate::metal::post::build_rt_reflection_pipeline;
use crate::metal::raytrace::{
    RtGpu, RtPipelines, RtSceneGeometry, RtState, RtStaticGeometry, RtTextureCounts,
    build_rt_accel, build_rt_skin_pipeline, raytracing_supported,
};
use crate::metal::slang_builtins::{RT_REFLECTIONS_FRAG, RT_REFLECTIONS_FRAG_TEXTURED};

pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) scene: &'a MtlSceneAssets,
    pub(super) glass: &'a GlassState,
    pub(super) post: &'a PostSettings,
}

pub(super) fn build_ray_tracing(gpu: &InitGpu<'_>, inputs: RtInputs<'_>) -> RenderResult<RtState> {
    let RtInputs {
        world,
        scene,
        glass,
        post,
    } = inputs;
    let device = &*gpu.hw.device;
    let pipelines = build_rt_pipelines(device, &post.rt_reflections, gpu.hot_reload)?;

    // The Layer 2 path is enabled when at least one material opts into
    // see-through AND the mesh pipeline built (RT-capable device). Mirrors
    // `MtlContext::seethrough_meshes_enabled`; the BVH must exclude the
    // see-through meshes it will reroute.
    let seethrough_enabled =
        !glass.seethrough_mesh_indices.is_empty() && glass.mesh_pipeline_rt.is_some();

    // Build the scene acceleration structure for hardware ray-traced
    // reflections. Only when the world enabled RT and the GPU supports ray
    // tracing; `build_rt_accel` returns None when the scene has no resident
    // geometry, in which case the RT pass stays a no-op (draw/mod gates
    // `rt_reflections_enabled` on this being Some). It needs the shared
    // geometry buffers + draw list; resolution-independent, so untouched on
    // resize. A wholesale rebuild on geometry change is the current update path.
    let accel = if post.rt_reflections.is_some() && raytracing_supported(device) {
        match build_rt_accel(
            RtGpu {
                device,
                command_queue: &gpu.hw.command_queue,
                frames_in_flight: gpu.frames_in_flight,
            },
            RtStaticGeometry {
                vertex_buffer: &scene.vertex_buffer,
                index_buffer: &scene.index_buffer,
            },
            RtSceneGeometry {
                draw_objects: &world.draw_objects,
                clusters: &world.instanced_clusters,
            },
            RtTextureCounts {
                albedo_count: scene.textures.len(),
            },
            // Skinned meshes upload after `new`, so the initial BVH is
            // static + instanced; the first frame's update seeds the
            // skinned geometry once `upload_skinned` has run.
            None,
            seethrough_enabled,
        )? {
            Some(a) => {
                tracing::info!(
                    "ray-traced reflections: built BVH over {} static objects",
                    a.blas.len()
                );
                Some(a)
            }
            None => {
                tracing::warn!(
                    "ray-traced reflections requested but the scene has no static geometry to build a BVH from; reflections disabled"
                );
                None
            }
        }
    } else {
        if post.rt_reflections.is_some() {
            tracing::warn!(
                "ray-traced reflections requested but this GPU does not support hardware ray tracing; falling back (no RT reflections)"
            );
        }
        None
    };

    if accel.is_some() {
        tracing::info!(
            "ray-traced reflections: dynamic transform mode = {:?}",
            post.rt_dynamic
        );
    }

    Ok(RtState {
        settings: post.rt_reflections,
        accel,
        dynamic_mode: post.rt_dynamic,
        skinned_geometry: post.rt_skinned_geometry,
        update_failed: false,
        topology_dirty: false,
        pipelines,
    })
}

// RT reflections: the inline ray-trace resolve pipelines, built only when RT
// reflections are on. They write into `ssr.targets.output`, reusing the SSR
// pre-pass G-buffer. The flat variant is the non-bindless fallback; the textured
// variant samples the bindless albedo pool. The compute-skinning pipeline
// deforms skinned vertices into a buffer the BVH can trace; built under the
// same gate on a ray-tracing device, unused when the world has no SkinnedMesh.
pub(in crate::metal) fn build_rt_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    settings: &Option<RtReflectionSettings>,
    hot_reload: bool,
) -> RenderResult<RtPipelines> {
    let (resolve, resolve_textured) = if settings.is_some() {
        (
            Some(build_rt_reflection_pipeline(
                device,
                &RT_REFLECTIONS_FRAG,
                hot_reload,
            )?),
            Some(build_rt_reflection_pipeline(
                device,
                &RT_REFLECTIONS_FRAG_TEXTURED,
                hot_reload,
            )?),
        )
    } else {
        (None, None)
    };
    let skin = if settings.is_some() && raytracing_supported(device) {
        Some(build_rt_skin_pipeline(device, hot_reload)?)
    } else {
        None
    };
    Ok(RtPipelines {
        resolve,
        resolve_textured,
        skin,
    })
}

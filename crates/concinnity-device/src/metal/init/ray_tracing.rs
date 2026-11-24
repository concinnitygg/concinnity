//! Hardware ray-traced reflections: the resolve and skinning pipelines, and the
//! scene acceleration structure built over the uploaded geometry.

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::init::{RenderLaunch, SceneData};
use concinnity_engine::render::post::rt_reflections::RtReflectionSettings;
use concinnity_engine::render::render_graph::PassQueue;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::InitGpu;
use crate::metal::blas_residency::BlasResidency;
use crate::metal::context::{GlassState, SceneAssets};
use crate::metal::post::RtReflectionPipelines;
use crate::metal::raytrace::{
    RtGpu, RtPipelines, RtSceneGeometry, RtState, RtStaticGeometry, RtTextureCounts,
    build_rt_accel, build_rt_skin_pipeline, raytracing_supported,
};

#[derive(Clone, Copy)]
pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) scene: &'a SceneAssets,
    pub(super) glass: &'a GlassState,
    pub(super) rt_reflections: Option<RtReflectionSettings>,
    pub(super) launch: RenderLaunch,
}

pub(super) fn build_ray_tracing(gpu: &InitGpu<'_>, inputs: RtInputs<'_>) -> RenderResult<RtState> {
    let RtInputs {
        world,
        scene,
        glass,
        rt_reflections,
        launch,
    } = inputs;
    let device = &*gpu.hw.device;
    let pipelines = build_rt_pipelines(device, rt_reflections.as_ref(), gpu.hot_reload)?;

    // The Layer 2 path is enabled when at least one material opts into
    // see-through AND the mesh pipeline built (RT-capable device). Mirrors
    // `MtlContext::seethrough_meshes_enabled`; the BVH must exclude the
    // see-through meshes it will reroute.
    let seethrough_enabled =
        !glass.seethrough_mesh_indices.is_empty() && glass.mesh_pipeline_rt.is_some();

    // The acceleration structure, when RT is on and supported. `None` with no
    // resident geometry. Resolution-independent.
    let accel = if rt_reflections.is_some() && raytracing_supported(device) {
        if let Some(a) = build_rt_accel(
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
            tracing::info!(
                "ray-traced reflections: built BVH over {} static objects",
                a.blas().len()
            );
            Some(a)
        } else {
            tracing::warn!(
                "ray-traced reflections requested but the scene has no static geometry to build a BVH from; reflections disabled"
            );
            None
        }
    } else {
        if rt_reflections.is_some() {
            tracing::error!(
                "the resolved settings asked for ray-traced reflections the device cannot run; \
                 the capability probe and the backend disagree"
            );
        }
        None
    };

    if accel.is_some() {
        tracing::info!(
            "ray-traced reflections: dynamic transform mode = {:?}",
            launch.rt_dynamic
        );
    }

    // Every queue a trace pass records on keeps the traced BLAS resident. Built
    // whatever the launch asked, since RT can be turned on later.
    let graphics = &*gpu.hw.command_queue;
    let async_compute = gpu
        .hw
        .graph_queues
        .as_ref()
        .map(|q| q.queue(PassQueue::AsyncCompute, graphics));
    let queues: Vec<_> = core::iter::once(graphics).chain(async_compute).collect();
    let blas_residency = raytracing_supported(device)
        .then(|| BlasResidency::new(device, &queues))
        .flatten();

    Ok(RtState {
        settings: rt_reflections,
        accel,
        blas_residency,
        dynamic_mode: launch.rt_dynamic,
        skinned_geometry: launch.rt_skinned_geometry,
        update_streak: concinnity_engine::render::rt_accel::FailureStreak::default(),
        pipelines,
        rays: None,
        retired: concinnity_engine::render::buffers::retire_pool::RetirePool::new(),
    })
}

// RT reflection kernels (classify plus flat and textured traces) and the
// skinning pipeline the BVH traces, built only with RT reflections on.
pub(in crate::metal) fn build_rt_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    settings: Option<&RtReflectionSettings>,
    hot_reload: bool,
) -> RenderResult<RtPipelines> {
    let reflections = settings
        .map(|_| RtReflectionPipelines::new(device, hot_reload))
        .transpose()?;
    let skin = if settings.is_some() && raytracing_supported(device) {
        Some(build_rt_skin_pipeline(device, hot_reload)?)
    } else {
        None
    };
    Ok(RtPipelines { reflections, skin })
}

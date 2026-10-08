// World Shader and SdfVolume pipelines built on a hot-reload worker.
//
// `vkCreateGraphicsPipelines` may run on any thread, and the device's pipeline
// cache is internally synchronized. What a worker cannot own is the render
// passes and layouts a pipeline is created against: they are raw handles the
// context destroys when it tears down. The builder therefore passes a
// `PipelineGate` the context closes before that teardown, and every build
// holds the gate open for as long as it reads the handles.

use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard};

use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;

use super::owned::VkDevice;
use super::pipeline::{BucketPipelineTargets, BucketPipelines, build_world_shader_pipeline};
use super::raymarch::{VolumePipelineTargets, VolumePipelines, build_volume_pipelines};

// Whether the context that handed out builders still holds the handles they
// build against. Clones share one gate, and it identifies its context: a
// pipeline built through another context's gate is never swapped in here.
#[derive(Clone, Default)]
pub(super) struct PipelineGate(Arc<RwLock<bool>>);

impl PipelineGate {
    // Hold the gate open for one build, or fail when the context has closed it.
    fn enter(&self) -> RenderResult<RwLockReadGuard<'_, bool>> {
        let closed = self.0.read().unwrap_or_else(PoisonError::into_inner);
        if *closed {
            return Err(RenderError::Other(
                "the renderer this pipeline was built for has shut down".into(),
            ));
        }
        Ok(closed)
    }

    // Close the gate, waiting out every build in flight. No build starts after.
    pub(super) fn close(&self) {
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = true;
    }

    fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub(super) struct VkPipelineBuilder {
    pub device: VkDevice,
    pub gate: PipelineGate,
    // `None` when the GPU-driven main pass is not live.
    pub world: Option<BucketPipelineTargets>,
    // `None` when the world has no raymarch pass.
    pub volumes: Option<VolumePipelineTargets>,
}

// A world Shader's main-pass and pre-pass pipelines and what they were built
// for.
struct PreparedWorldShader {
    gate: PipelineGate,
    targets: BucketPipelineTargets,
    pipelines: BucketPipelines,
}

// A volume's pipelines and what they were built for.
struct PreparedVolume {
    gate: PipelineGate,
    targets: VolumePipelineTargets,
    flags: VolumeFlags,
    pipelines: VolumePipelines,
}

impl PipelineBuilder for VkPipelineBuilder {
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        let targets = self
            .world
            .ok_or_else(|| RenderError::Other("the GPU-driven main pass is not live".into()))?;
        let _open = self.gate.enter()?;
        let pipelines =
            build_world_shader_pipeline(&self.device, targets, bucket as usize, programs)?;
        Ok(PreparedPipelines::new(PreparedWorldShader {
            gate: self.gate.clone(),
            targets,
            pipelines,
        }))
    }

    fn sdf_volume(
        &self,
        programs: &SdfPrograms,
        flags: VolumeFlags,
        label: &str,
    ) -> RenderResult<PreparedPipelines> {
        let targets = self
            .volumes
            .ok_or_else(|| RenderError::Other("the world has no raymarch pass".into()))?;
        let _open = self.gate.enter()?;
        let pipelines = build_volume_pipelines(&self.device, &targets, programs, flags, label)?;
        Ok(PreparedPipelines::new(PreparedVolume {
            gate: self.gate.clone(),
            targets,
            flags,
            pipelines,
        }))
    }
}

// The world Shader pipelines in `prepared` when this context's `gate` built
// them for `targets`.
pub(super) fn world_shader_for(
    prepared: Option<PreparedPipelines>,
    gate: &PipelineGate,
    targets: BucketPipelineTargets,
) -> Option<BucketPipelines> {
    prepared?
        .downcast::<PreparedWorldShader>()
        .filter(|p| p.gate.is(gate) && p.targets == targets)
        .map(|p| p.pipelines)
}

// The volume pipelines in `prepared` when this context's `gate` built them for
// a volume with `flags` against `targets`.
pub(super) fn volume_for(
    prepared: Option<PreparedPipelines>,
    gate: &PipelineGate,
    targets: VolumePipelineTargets,
    flags: VolumeFlags,
) -> Option<VolumePipelines> {
    prepared?
        .downcast::<PreparedVolume>()
        .filter(|p| p.gate.is(gate) && p.targets == targets && p.flags == flags)
        .map(|p| p.pipelines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    // A prepared pipeline matches only the template reload it was built from,
    // so one a worker finished after a reload is rebuilt instead.
    #[test]
    fn targets_differ_across_a_template_reload() {
        let targets = |template_generation| BucketPipelineTargets {
            render_pass: vk::RenderPass::null(),
            layout: vk::PipelineLayout::null(),
            prepass: None,
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            swapchain_format: vk::Format::B8G8R8A8_UNORM,
            hot_reload: true,
            template_generation,
        };
        assert!(targets(1) == targets(1));
        assert!(targets(1) != targets(2));
    }

    #[test]
    fn a_closed_gate_refuses_every_later_build() {
        let gate = PipelineGate::default();
        assert!(gate.enter().is_ok());
        gate.clone().close();
        assert!(gate.enter().is_err());
    }

    // Closing waits for a build that holds the gate open.
    #[test]
    fn closing_waits_for_a_build_in_flight() {
        let gate = PipelineGate::default();
        let open = gate.enter().unwrap();
        let closer = gate.clone();
        let (done, closed) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            closer.close();
            done.send(()).unwrap();
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(closed.try_recv().is_err(), "closed under an open build");
        drop(open);
        closed
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the gate closed once the build ended");
        thread.join().unwrap();
    }

    #[test]
    fn a_gate_is_only_its_own_clones() {
        let gate = PipelineGate::default();
        assert!(gate.is(&gate.clone()));
        assert!(!gate.is(&PipelineGate::default()));
    }
}

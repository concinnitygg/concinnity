// World Shader and SdfVolume pipelines built on a hot-reload worker.
//
// `ID3D12Device` is free-threaded and every object a PSO is created against is
// reference counted, so the builder holds its own references to the device and
// the root signatures. A PSO built against a root signature the context no
// longer binds is never swapped in: the context compares the one it was built
// against with its own. The pipeline library is the frame thread's, so a
// worker's PSO builds uncached.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12InfoQueue, ID3D12PipelineState, ID3D12RootSignature,
};

use super::init::pipelines::{WorldPsoTargets, build_world_shader_pso};
use super::raymarch::{VolumePsoTargets, VolumePsos, build_volume_psos};

// The raymarch pass's two root signatures, which every volume PSO binds.
#[derive(Clone, PartialEq)]
pub(super) struct VolumeRootSigs {
    pub root_sig: ID3D12RootSignature,
    pub shadow_root_sig: ID3D12RootSignature,
}

// What a pipeline is built against beyond the device. A prepared pipeline is
// swapped in only while the context still matches the targets it was built for.
#[derive(Clone, PartialEq)]
pub(super) struct Targets<R> {
    pub root_sigs: R,
    pub msaa_samples: u32,
    pub hot_reload: bool,
}

pub(super) struct DxPipelineBuilder {
    pub device: ID3D12Device,
    pub info_queue: Option<ID3D12InfoQueue>,
    // `None` when the GPU-driven main pass is not live.
    pub world: Option<Targets<ID3D12RootSignature>>,
    // `None` when the world declares no volume.
    pub volumes: Option<Targets<VolumeRootSigs>>,
}

// A world Shader's main-pass PSO and what it was built for.
pub(super) struct PreparedWorldShader {
    pub targets: Targets<ID3D12RootSignature>,
    pub pso: ID3D12PipelineState,
}

// A volume's PSOs and what they were built for.
pub(super) struct PreparedVolume {
    pub targets: Targets<VolumeRootSigs>,
    pub flags: VolumeFlags,
    pub psos: VolumePsos,
}

impl PipelineBuilder for DxPipelineBuilder {
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        let targets = self
            .world
            .clone()
            .ok_or_else(|| RenderError::Other("the GPU-driven main pass is not live".into()))?;
        let pso = build_world_shader_pso(
            &self.device,
            self.info_queue.as_ref(),
            WorldPsoTargets {
                root_sig: &targets.root_sigs,
                msaa_samples: targets.msaa_samples,
                hot_reload: targets.hot_reload,
            },
            bucket as usize,
            programs,
        )?;
        Ok(PreparedPipelines::new(PreparedWorldShader { targets, pso }))
    }

    fn sdf_volume(
        &self,
        programs: &SdfPrograms,
        flags: VolumeFlags,
        label: &str,
    ) -> RenderResult<PreparedPipelines> {
        let targets = self
            .volumes
            .clone()
            .ok_or_else(|| RenderError::Other("the world has no raymarch pass".into()))?;
        let psos = build_volume_psos(
            &VolumePsoTargets {
                device: &self.device,
                info_queue: self.info_queue.as_ref(),
                root_sig: &targets.root_sigs.root_sig,
                shadow_root_sig: &targets.root_sigs.shadow_root_sig,
                msaa_samples: targets.msaa_samples,
                hot_reload: targets.hot_reload,
            },
            programs,
            flags,
            label,
        )?;
        Ok(PreparedPipelines::new(PreparedVolume {
            targets,
            flags,
            psos,
        }))
    }
}

// The world Shader PSO in `prepared` when it was built for `targets`.
pub(super) fn world_shader_for(
    prepared: Option<PreparedPipelines>,
    targets: &Targets<ID3D12RootSignature>,
) -> Option<ID3D12PipelineState> {
    prepared?
        .downcast::<PreparedWorldShader>()
        .filter(|p| p.targets == *targets)
        .map(|p| p.pso)
}

// The volume PSOs in `prepared` when they were built for a volume with `flags`
// against `targets`.
pub(super) fn volume_for(
    prepared: Option<PreparedPipelines>,
    targets: &Targets<VolumeRootSigs>,
    flags: VolumeFlags,
) -> Option<VolumePsos> {
    prepared?
        .downcast::<PreparedVolume>()
        .filter(|p| p.targets == *targets && p.flags == flags)
        .map(|p| p.psos)
}

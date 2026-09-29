// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, handing over a pipeline its worker already built.
//
// The bucket regions of the GPU-culled command buffer are issued here too: the
// cull kernel wrote every record's command into exactly one region, so each
// region is one `ExecuteIndirect` under that bucket's pipeline. Mirrors
// `metal/world_shaders.rs` + the bucket loop in `metal/draw/main.rs`.
//
// Unlike Metal there is no on-disk GPU-binary cache behind the build: see
// `docs/todos.md` for why the D3D12 pipeline-library equivalent is still open.

use concinnity_core::render::backend::{PipelineBuilder, PipelineSwap, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;

use super::context::DxContext;
use super::init::pipelines::{WorldPsoTargets, build_world_shader_pso};
use super::pipeline_builder::{DxPipelineBuilder, Targets, VolumeRootSigs, world_shader_for};

impl DxContext {
    // Install one shader bucket's bindless main-pass pipeline: `prepared` when
    // it was built for this context's targets, else one built here. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // installs cleanly.
    pub(in crate::directx) fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        let slot = self.world_pipeline_slot(bucket)?;
        let targets = self.world_pso_targets().ok_or_else(|| {
            RenderError::Other("shader buckets need the bindless main pass".into())
        })?;
        let pso = match world_shader_for(prepared, &targets) {
            Some(pso) => pso,
            None => build_world_shader_pso(
                &self.hw.device,
                self.hw.info_queue.as_ref(),
                WorldPsoTargets {
                    root_sig: &targets.root_sigs,
                    msaa_samples: targets.msaa_samples,
                    hot_reload: targets.hot_reload,
                },
                bucket as usize,
                programs,
            )?,
        };
        self.evict_world_shader(bucket);
        self.cull.world_pipelines[slot] = Some(pso);
        Ok(())
    }

    // Rebuild one world Shader's pipeline from hot-reloaded programs, or swap
    // in `prepared` when it was built for this context's targets. Bucket 0 is
    // the main pass's PSO; another bucket is rebuilt only while installed, and
    // the replacement is built before the old PSO retires, so a failed build
    // leaves the live one bound.
    pub(in crate::directx) fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if bucket == 0 {
            let prepared = self
                .world_pso_targets()
                .and_then(|targets| world_shader_for(prepared, &targets));
            self.update_default_world_shader(programs, prepared)?;
            return Ok(PipelineSwap::Swapped);
        }
        self.world_pipeline_slot(bucket)?;
        if !self.world_shader_resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        self.install_world_shader(bucket, programs, prepared)?;
        Ok(PipelineSwap::Swapped)
    }

    // What a world Shader's PSO is built against here, or `None` when the
    // GPU-driven main pass is not live.
    pub(in crate::directx) fn world_pso_targets(&self) -> Option<Targets<ID3D12RootSignature>> {
        Some(Targets {
            root_sigs: self.cull.main_bindless_root_sig.clone()?,
            msaa_samples: self.targets.hdr.msaa_samples,
            hot_reload: self.hot_reload.enabled,
        })
    }

    // What a volume's PSOs are built against here, or `None` when the world
    // has no raymarch pass.
    pub(in crate::directx) fn volume_pso_targets(&self) -> Option<Targets<VolumeRootSigs>> {
        let rm = self.raymarch.as_ref()?;
        Some(Targets {
            root_sigs: VolumeRootSigs {
                root_sig: rm.root_sig.clone(),
                shadow_root_sig: rm.shadow_root_sig.clone(),
            },
            msaa_samples: self.targets.hdr.msaa_samples,
            hot_reload: self.hot_reload.enabled,
        })
    }

    // A builder for this context's world Shader and volume PSOs, for a
    // streaming or hot-reload worker.
    pub(in crate::directx) fn pipeline_builder(&self) -> Arc<dyn PipelineBuilder> {
        Arc::new(DxPipelineBuilder {
            device: self.hw.device.clone(),
            info_queue: self.hw.info_queue.clone(),
            world: self.world_pso_targets(),
            volumes: self.volume_pso_targets(),
        })
    }

    // Release one bucket's pipeline. D3D12 command lists do not keep a pipeline
    // state alive, so the PSO is retired through the allocator's frame tick,
    // which holds it until every frame in flight that recorded against it has
    // finished.
    pub(in crate::directx) fn evict_world_shader(&mut self, bucket: u32) {
        let Ok(slot) = self.world_pipeline_slot(bucket) else {
            return;
        };
        if let Some(pso) = self.cull.world_pipelines[slot].take() {
            self.hw.alloc.retire(pso);
        }
    }

    // Whether a bucket's draws can render this frame: bucket 0 is the world
    // default program, every other bucket needs its pipeline installed.
    pub(in crate::directx) fn world_shader_resident(&self, bucket: usize) -> bool {
        bucket == 0
            || matches!(
                self.cull.world_pipelines.get(bucket.wrapping_sub(1)),
                Some(Some(_))
            )
    }

    pub(in crate::directx) fn world_pipeline(&self, bucket: usize) -> Option<&ID3D12PipelineState> {
        self.cull
            .world_pipelines
            .get(bucket.checked_sub(1)?)?
            .as_ref()
    }

    // Issue the bucket 1.. regions of `indirect`, each under its own material
    // shader's pipeline. Bucket 0's region is issued by the caller (it runs under
    // the pipeline the pass already bound), so this covers only the
    // material-referenced shaders. `max_count` is the record prefix each region
    // draws, matching bucket 0's. Returns the number of `ExecuteIndirect` calls
    // issued, and leaves the last bucket's pipeline bound.
    pub(in crate::directx) fn execute_bucket_regions(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        cull_sig: &ID3D12CommandSignature,
        indirect: &ID3D12Resource,
        max_count: u32,
    ) -> u32 {
        self.for_each_resident_bucket(|bucket| {
            let Some(pso) = self.world_pipeline(bucket) else {
                return;
            };
            // Every bucket shares the bindless root signature, so the Wireframe
            // twin stands in for each one while that view mode is on.
            let pso = self.wireframe_or(pso, self.wireframe.bindless.as_ref());
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.SetPipelineState(pso) };
            self.execute_bucket_region(cmd, cull_sig, indirect, max_count, bucket);
        })
    }

    // Issue the bucket 1.. regions of `indirect` under the pipeline the caller
    // already bound. Used by the depth / velocity pre-pass, which shades nothing
    // and so runs every bucket through its own single pipeline -- but still has to
    // skip a non-resident bucket, or the pre-pass would lay down depth and motion
    // for geometry the color pass omits.
    pub(in crate::directx) fn execute_bucket_regions_shared_pso(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        cull_sig: &ID3D12CommandSignature,
        indirect: &ID3D12Resource,
        max_count: u32,
    ) -> u32 {
        self.for_each_resident_bucket(|bucket| {
            self.execute_bucket_region(cmd, cull_sig, indirect, max_count, bucket)
        })
    }

    // Run `f` for every bucket past the default whose Shader is resident,
    // returning how many ran. A bucket whose scene has not pinned yet has no
    // pipeline: skip it until warmup builds one rather than drawing it with the
    // wrong program.
    fn for_each_resident_bucket(&self, mut f: impl FnMut(usize)) -> u32 {
        let mut issued = 0;
        for bucket in 1..self.shader_bucket_count() {
            if !self.world_shader_resident(bucket) {
                continue;
            }
            f(bucket);
            issued += 1;
        }
        issued
    }

    fn execute_bucket_region(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        cull_sig: &ID3D12CommandSignature,
        indirect: &ID3D12Resource,
        max_count: u32,
        bucket: usize,
    ) {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ExecuteIndirect(
                cull_sig,
                max_count,
                indirect,
                self.bucket_region_offset(bucket),
                None::<&ID3D12Resource>,
                0,
            );
        }
    }

    fn world_pipeline_slot(&self, bucket: u32) -> RenderResult<usize> {
        let slot = (bucket as usize).checked_sub(1).ok_or_else(|| {
            RenderError::Other("shader bucket 0 is the world default program".into())
        })?;
        if slot >= self.cull.world_pipelines.len() {
            return Err(RenderError::Other(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.cull.world_pipelines.len()
            )));
        }
        Ok(slot)
    }
}

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
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::world_pipelines::{check_rebuild, replace_bucket};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;

use super::context::DxContext;
use super::init::pipelines::{
    BindlessMainShaders, BucketPipelineTargets, BucketPsos, BucketRootSigs, PrepassSource,
    WorldPsoTargets, build_bucket_pipeline, build_prepass_for, build_world_shader_pso,
};
use super::pipeline_builder::{DxPipelineBuilder, Targets, VolumeRootSigs, world_shader_for};

impl DxContext {
    // Install one shader bucket's main-pass and pre-pass PSOs: `prepared` when
    // they were built for this context's targets, else ones built here.
    // Replaces whatever the bucket currently holds, so a re-pin after an
    // eviction installs cleanly.
    pub(in crate::directx) fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<()> {
        let psos = self.bucket_psos(bucket, programs, prepared, false)?;
        self.install_bucket(bucket, psos)
    }

    // One shader bucket's PSOs to replace a bucket that has a pre-pass when
    // `live_prepass`: `prepared` when it keeps what the live bucket draws with,
    // else ones built here. Refused when the build would drop a live pre-pass,
    // so the caller keeps the live pair.
    fn bucket_psos(
        &self,
        bucket: u32,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<PreparedPipelines>,
        live_prepass: bool,
    ) -> RenderResult<BucketPsos> {
        self.cull.world_pipelines.slot(bucket)?;
        let targets = self.world_pso_targets().ok_or_else(|| {
            RenderError::Other("shader buckets need the bindless main pass".into())
        })?;
        replace_bucket(
            bucket as usize,
            live_prepass,
            world_shader_for(prepared, &targets),
            |p: &BucketPsos| p.prepass.is_some(),
            || {
                build_world_shader_pso(
                    &self.hw.device,
                    self.hw.info_queue.as_ref(),
                    WorldPsoTargets {
                        root_sigs: &targets.root_sigs,
                        msaa_samples: targets.msaa_samples,
                        hot_reload: targets.hot_reload,
                    },
                    bucket as usize,
                    programs,
                )
            },
        )
    }

    fn install_bucket(&mut self, bucket: u32, psos: BucketPsos) -> RenderResult<()> {
        if let Some(displaced) = self.cull.world_pipelines.install(bucket, psos)? {
            self.retire_bucket(displaced);
        }
        Ok(())
    }

    // Retire both of a displaced bucket's PSOs through the allocator's frame
    // tick, which holds each until every frame in flight that recorded against
    // it has finished.
    fn retire_bucket(&self, psos: BucketPsos) {
        self.hw.alloc.retire(psos.main);
        if let Some(prepass) = psos.prepass {
            self.hw.alloc.retire(prepass);
        }
    }

    // Build the pre-pass PSO each bucket is missing while a G-buffer exists. A
    // PSO that fails to build leaves its bucket without G-buffer draws, never
    // without shading. The G-buffer is never torn down once built, so there
    // is nothing to drop here.
    pub(in crate::directx) fn sync_prepass_psos(&mut self) {
        let (Some(_), Some(root_sig)) = (&self.gbuffer, self.cull.prepass_root_sig.clone()) else {
            return;
        };
        let device = self.hw.device.clone();
        let info_queue = self.hw.info_queue.clone();
        let hot_reload = self.hot_reload.enabled;
        let build = |bucket: usize, source: PrepassSource<'_>| {
            build_prepass_for(&device, info_queue.as_ref(), &root_sig, bucket, source)
        };
        if self.cull.main_bindless_pso.is_some() && self.cull.main_prepass_pso.is_none() {
            let source = match self.world_shader.as_ref() {
                Some(programs) => PrepassSource::World(programs, hot_reload),
                None => PrepassSource::Engine(&self.cull.bindless_main_shaders),
            };
            self.cull.main_prepass_pso = build(0, source);
        }
        let buckets: Vec<usize> = self.cull.world_pipelines.resident_buckets().collect();
        for bucket in buckets {
            let Some(psos) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            if psos.prepass.is_some() {
                continue;
            }
            let fresh = match psos.programs.as_ref() {
                Some(programs) => build(bucket, PrepassSource::World(programs, hot_reload)),
                None => build(
                    bucket,
                    PrepassSource::Engine(&self.cull.bindless_main_shaders),
                ),
            };
            if let Some(psos) = self.cull.world_pipelines.get_mut(bucket) {
                psos.prepass = fresh;
            }
        }
    }

    // Every resident material bucket's PSOs rebuilt from the current
    // templates, `engine` standing in for a bucket with no Shader of its own.
    // Fails as a whole when any bucket's main PSO fails or its rebuild would
    // drop a live pre-pass, so the caller keeps every live pair.
    pub(in crate::directx) fn rebuild_world_buckets(
        &self,
        engine: &BindlessMainShaders,
    ) -> RenderResult<Vec<(usize, BucketPsos)>> {
        let Some(targets) = self.world_pso_targets() else {
            return Ok(Vec::new());
        };
        let mut rebuilt = Vec::new();
        for bucket in self.cull.world_pipelines.resident_buckets() {
            let Some(live) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            let psos = build_bucket_pipeline(
                &self.hw.device,
                self.hw.info_queue.as_ref(),
                BucketPipelineTargets {
                    root_sigs: &targets.root_sigs,
                    msaa_samples: targets.msaa_samples,
                    engine_default: engine,
                    hot_reload: targets.hot_reload,
                },
                bucket,
                WorldShader {
                    programs: live.programs.as_ref(),
                    deferred: false,
                },
            )?;
            check_rebuild(bucket, live.prepass.is_some(), psos.prepass.is_some())?;
            rebuilt.push((bucket, psos));
        }
        Ok(rebuilt)
    }

    // Swap in `rebuild_world_buckets`' PSOs, retiring the ones they replace.
    pub(in crate::directx) fn swap_world_buckets(&mut self, rebuilt: Vec<(usize, BucketPsos)>) {
        let mut displaced = Vec::new();
        for (bucket, psos) in rebuilt {
            if let Some(live) = self.cull.world_pipelines.get_mut(bucket) {
                displaced.push(std::mem::replace(live, psos));
            }
        }
        for psos in displaced {
            self.retire_bucket(psos);
        }
    }

    // Rebuild one world Shader's PSOs from hot-reloaded programs, or swap in
    // `prepared` when they were built for this context's targets. Bucket 0 is
    // the main pass's and the pre-pass's own pair; another bucket is rebuilt
    // only while installed. Both halves are built before the old PSOs retire,
    // so a failed build of either leaves the live ones bound.
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
        self.cull.world_pipelines.slot(bucket)?;
        if !self.cull.world_pipelines.resident(bucket as usize) {
            return Ok(PipelineSwap::NotResident);
        }
        let live_prepass = (self.cull.world_pipelines.get(bucket as usize))
            .is_some_and(|live| live.prepass.is_some());
        let psos = self.bucket_psos(bucket, programs, prepared, live_prepass)?;
        self.install_bucket(bucket, psos)?;
        Ok(PipelineSwap::Swapped)
    }

    // What a world Shader's PSOs are built against here, or `None` when the
    // GPU-driven main pass is not live.
    pub(in crate::directx) fn world_pso_targets(&self) -> Option<Targets<BucketRootSigs>> {
        Some(Targets {
            root_sigs: BucketRootSigs {
                main: self.cull.main_bindless_root_sig.clone()?,
                prepass: self
                    .cull
                    .prepass_root_sig
                    .clone()
                    .filter(|_| self.gbuffer.is_some()),
            },
            msaa_samples: self.targets.hdr.msaa_samples,
            hot_reload: self.hot_reload.enabled,
            template_generation: self.hot_reload.generation,
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
            template_generation: self.hot_reload.generation,
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

    // Release one bucket's pipelines. D3D12 command lists do not keep a
    // pipeline state alive, so each PSO is retired through the allocator's frame
    // tick, which holds it until every frame in flight that recorded against it
    // has finished.
    pub(in crate::directx) fn evict_world_shader(&mut self, bucket: u32) {
        if let Some(psos) = self.cull.world_pipelines.evict(bucket) {
            self.retire_bucket(psos);
        }
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
            let Some(psos) = self.cull.world_pipelines.get(bucket) else {
                return;
            };
            // Every bucket shares the bindless root signature, so the Wireframe
            // twin stands in for each one while that view mode is on.
            let pso = self.wireframe_or(&psos.main, self.wireframe.bindless.as_ref());
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.SetPipelineState(pso) };
            self.execute_bucket_region(cmd, cull_sig, indirect, max_count, bucket);
        })
    }

    // Issue the bucket 1.. regions of `indirect`, each under its shader's
    // G-buffer pre-pass PSO, so a world Shader's vertex hook places its depth
    // and motion as it places its shading. A non-resident bucket is skipped
    // here as in the main pass, so the pre-pass never lays down depth and motion
    // for geometry the color pass omits, and so is one whose pre-pass PSO
    // failed to build. Returns the regions issued, and leaves the last
    // bucket's PSO bound.
    pub(in crate::directx) fn execute_prepass_bucket_regions(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        prepass_sig: &ID3D12CommandSignature,
        indirect: &ID3D12Resource,
        max_count: u32,
    ) -> u32 {
        let mut issued = 0;
        for bucket in self.cull.world_pipelines.resident_buckets() {
            let Some(prepass) = self
                .cull
                .world_pipelines
                .get(bucket)
                .and_then(|psos| psos.prepass.as_ref())
            else {
                continue;
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe { cmd.SetPipelineState(prepass) };
            self.execute_bucket_region(cmd, prepass_sig, indirect, max_count, bucket);
            issued += 1;
        }
        issued
    }

    // Run `f` for every bucket past the default whose Shader is resident,
    // returning how many ran. A bucket whose scene has not pinned yet has no
    // pipeline: skip it until warmup builds one rather than drawing it with the
    // wrong program.
    fn for_each_resident_bucket(&self, mut f: impl FnMut(usize)) -> u32 {
        let mut issued = 0;
        for bucket in self.cull.world_pipelines.resident_buckets() {
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
}

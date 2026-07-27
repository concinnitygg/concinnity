// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, so the pipeline build lands behind the loading
// screen rather than on the frame that first draws the material.
//
// The bucket regions of the GPU-culled command buffer are issued here too: the
// cull kernel wrote every record's command into exactly one region, so each
// region is one `ExecuteIndirect` under that bucket's pipeline. Mirrors
// `metal/world_shaders.rs` + the bucket loop in `metal/draw/main.rs`.
//
// Unlike Metal there is no on-disk GPU-binary cache behind the build: see
// `docs/todos.md` for why the D3D12 pipeline-library equivalent is still open.

use windows::Win32::Graphics::Direct3D12::*;

use super::context::DxContext;
use super::init::pipelines::build_bucket_pipeline;

impl DxContext {
    // Build the bindless main-pass pipeline for one shader bucket. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // rebuilds cleanly.
    pub(in crate::directx) fn install_world_shader(
        &mut self,
        bucket: u32,
        shader: crate::gfx::backend_init::ShaderBytes<'_>,
    ) -> Result<(), String> {
        let slot = self.world_pipeline_slot(bucket)?;
        let root_sig = self
            .cull
            .main_bindless_root_sig
            .clone()
            .ok_or_else(|| "shader buckets need the bindless main pass".to_string())?;
        let pso = build_bucket_pipeline(
            &self.device,
            self.info_queue.as_ref(),
            &root_sig,
            bucket as usize,
            shader,
            self.hdr.msaa_samples,
            &self.bindless_main_shaders,
        )?;
        // A re-pin over a slot that still holds a pipeline has the same in-flight
        // hazard as an evict, so retire the old one the same way.
        self.evict_world_shader(bucket);
        self.cull.world_pipelines[slot] = Some(pso);
        Ok(())
    }

    // Release one bucket's pipeline. D3D12 command lists do not keep a pipeline
    // state alive, so the in-flight frames that recorded against it have to
    // finish before the last reference drops -- otherwise the GPU reads a freed
    // pipeline and the device falls over. An evict only happens when a scene
    // unpins, which is already a loading-screen stall, so draining the queue here
    // costs nothing a player sees.
    pub(in crate::directx) fn evict_world_shader(&mut self, bucket: u32) {
        let Ok(slot) = self.world_pipeline_slot(bucket) else {
            return;
        };
        if self.cull.world_pipelines[slot].is_none() {
            return;
        }
        self.wait_idle();
        self.cull.world_pipelines[slot] = None;
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
            unsafe { cmd.SetPipelineState(pso) };
            self.execute_bucket_region(cmd, cull_sig, indirect, max_count, bucket);
        })
    }

    // Issue the bucket 1.. regions of `indirect` under the pipeline the caller
    // already bound. Used by the depth / velocity pre-pass, which shades nothing
    // and so runs every bucket through its own single pipeline -- but still has to
    // skip a non-resident bucket, or the pre-pass would lay down depth and motion
    // for geometry the colour pass omits.
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

    fn world_pipeline_slot(&self, bucket: u32) -> Result<usize, String> {
        let slot = (bucket as usize)
            .checked_sub(1)
            .ok_or_else(|| "shader bucket 0 is the world default program".to_string())?;
        if slot >= self.cull.world_pipelines.len() {
            return Err(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.cull.world_pipelines.len()
            ));
        }
        Ok(slot)
    }
}

// Runtime residency of the material-referenced world shader pipelines.
//
// Init builds a pipeline for every world Shader whose payload it decoded and
// leaves a `None` in `world_pipelines` for each one it deferred (a Shader owned
// by a scene other than the start scene). The streaming pump calls in here as
// those scenes pin and unpin, so the pipeline build lands behind the loading
// screen rather than on the frame that first draws the material.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLRenderPipelineState;

use super::MtlContext;
use super::init::pipelines::{build_bucket_pipeline, make_vertex_descriptor};

impl MtlContext {
    // Build the bindless main-pass pipeline for one shader bucket. Replaces
    // whatever the bucket currently holds, so a re-pin after an eviction
    // rebuilds cleanly.
    pub(super) fn install_world_shader(
        &mut self,
        bucket: u32,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
    ) -> Result<(), String> {
        let slot = self.world_pipeline_slot(bucket)?;
        let vert_desc = make_vertex_descriptor();
        let pso = build_bucket_pipeline(
            &self.device,
            &vert_desc,
            bucket as usize,
            vert_bytes,
            frag_bytes,
        )?;
        self.world_pipelines[slot] = Some(pso);
        Ok(())
    }

    // Release one bucket's pipeline. A Metal command buffer retains the
    // pipelines encoded into it, so dropping this reference mid-frame is safe
    // and the next frame's main pass simply skips the bucket. That retention is
    // Metal's alone: DirectX and Vulkan command lists do NOT keep a pipeline
    // alive, so their evict drains the device first.
    pub(super) fn evict_world_shader(&mut self, bucket: u32) {
        if let Ok(slot) = self.world_pipeline_slot(bucket) {
            self.world_pipelines[slot] = None;
        }
    }

    // Whether a bucket's draws can render this frame: bucket 0 is the world
    // default program, every other bucket needs its pipeline installed.
    pub(super) fn world_shader_resident(&self, bucket: usize) -> bool {
        bucket == 0
            || matches!(
                self.world_pipelines.get(bucket.wrapping_sub(1)),
                Some(Some(_))
            )
    }

    pub(super) fn world_pipeline(
        &self,
        bucket: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.world_pipelines.get(bucket.checked_sub(1)?)?.as_ref()
    }

    fn world_pipeline_slot(&self, bucket: u32) -> Result<usize, String> {
        let slot = (bucket as usize)
            .checked_sub(1)
            .ok_or_else(|| "shader bucket 0 is the world default program".to_string())?;
        if slot >= self.world_pipelines.len() {
            return Err(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.world_pipelines.len()
            ));
        }
        Ok(slot)
    }
}

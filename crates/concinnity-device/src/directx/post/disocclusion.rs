//! DirectX's share of the reactive mask's disocclusion term: the shared pass is
//! kept beside an upscaler that reads the mask, and each frame it reads the
//! G-buffer and writes the mask the graph has put in `RENDER_TARGET`. The mark
//! and the snapshot are written once in
//! `concinnity_engine::render::post::disocclusion`.

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::disocclusion::{DisocclusionInputs, DisocclusionPass};
use concinnity_engine::render::temporal::reactive_mask::ReactiveWrite;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::post::post_device::{DxAttachment, FullscreenExtent, PostPipeline, PostTarget};

// The shared pass, holding DirectX's own pipeline and target handles.
pub(in crate::directx) type DxDisocclusionPass = DisocclusionPass<PostPipeline, PostTarget>;

impl DxContext {
    // Build, resize or drop the pass to match the upscaler and the G-buffer at
    // the current render size. The caller has already idled the device.
    pub(in crate::directx) fn sync_disocclusion(&mut self) -> RenderResult<()> {
        let wanted = self.gbuffer.is_some()
            && self
                .upscale
                .backend
                .as_ref()
                .is_some_and(|u| u.reactive_reader().reads());
        let extent = PostExtent {
            width: self.targets.extent.render_width,
            height: self.targets.extent.render_height,
        };
        self.disocclusion = match (wanted, self.disocclusion.take()) {
            (false, _) => None,
            (true, Some(mut pass)) => {
                pass.resize(&self.post_device(0), extent)?;
                Some(pass)
            }
            (true, None) => Some(DisocclusionPass::new(&self.post_device(0), extent)?),
        };
        Ok(())
    }

    // Encode the mark into the reactive mask, then this frame's snapshot.
    pub(in crate::directx) fn encode_disocclusion(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) -> RenderResult<()> {
        let missing = |what: &str| RenderError::Other(format!("disocclusion: no {what}"));
        let pass = self.disocclusion.as_ref().ok_or_else(|| missing("pass"))?;
        let gbuffer = self.gbuffer.as_ref().ok_or_else(|| missing("G-buffer"))?;
        let mask = &self.targets.reactive_mask;
        pass.encode(
            &self.post_device(frame_idx),
            cmd,
            &DisocclusionInputs {
                normal_depth: gbuffer.normal_depth_srv_gpu,
                velocity: gbuffer.velocity_srv_gpu,
                mask: DxAttachment {
                    resource: &mask.resource,
                    subresource: 0,
                    rtv: mask.rtv(ReactiveWrite::Clear),
                    extent: FullscreenExtent {
                        width: self.targets.extent.render_width,
                        height: self.targets.extent.render_height,
                    },
                },
            },
        )
    }
}

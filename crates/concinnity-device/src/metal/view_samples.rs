//! View samples on Metal: the dispatch that samples the Hi-Z pyramid's first
//! level over a coarse grid of cells, and the ring of shared buffers the CPU
//! reads them back from.

use concinnity_engine::gfx::render_types::{VIEW_SAMPLE_CELLS, ViewSampleCell, ViewSampleParams};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::view::SAMPLE_GROUP;
use concinnity_engine::render::shader::programs::abi::view::metal::{CELLS, DEPTH, PARAMS};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLDevice, MTLTexture,
};

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::dispatch::dispatch_groups;
use super::encode::ComputeEncode;
use super::readback_ring::ReadbackRing;
use super::scoped_encoder::ScopedEncoder;

/// The sampling's pipeline and its readback ring.
pub(super) struct ViewSamplesGpu {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // One grid of cells per frame in flight; a slot no frame wrote reads back
    // as serial 0.
    outputs: ReadbackRing<ViewSampleCell>,
}

impl ViewSamplesGpu {
    /// The sampling with a readback slot per frame in flight.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
        frames_in_flight: usize,
    ) -> RenderResult<Self> {
        Ok(Self {
            pipeline: Self::build_pipeline(device, hot_reload)?,
            outputs: ReadbackRing::new(
                device,
                frames_in_flight,
                VIEW_SAMPLE_CELLS,
                "view samples",
            )?,
        })
    }

    /// The kernel, from `view_samples.hlsl`.
    pub(super) fn build_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        compute_pipeline(device, &super::builtin_shaders::VIEW_SAMPLES, hot_reload)
    }

    /// The cells ring slot `slot` holds, once the fence has retired its
    /// writer.
    pub(super) fn read(&self, slot: usize) -> Option<&[ViewSampleCell]> {
        self.outputs.read(slot)
    }

    // Encode the dispatch into `enc`: `depth` sampled by `params` into ring
    // slot `slot`.
    fn dispatch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        depth: &ProtocolObject<dyn MTLTexture>,
        params: &ViewSampleParams,
        slot: usize,
    ) {
        let Some(output) = self.outputs.buffer(slot) else {
            return;
        };
        enc.set_pipeline(&self.pipeline);
        enc.set_texture(depth, DEPTH.index);
        enc.set_buffer(output, 0, CELLS.index);
        enc.set_value(params, PARAMS.index);
        dispatch_groups(enc, [params.columns, params.rows, 1], SAMPLE_GROUP);
    }
}

impl MtlContext {
    // Sample the Hi-Z pyramid's first level by this frame's view parameters
    // into ring slot `slot`. A no-op without the sampling, the pyramid, or
    // samples planned this frame.
    pub(in crate::metal) fn encode_view_samples(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        slot: usize,
    ) -> RenderResult<u32> {
        let (Some(gpu), Some(hiz), Some(params)) = (
            self.view_samples.as_ref(),
            self.cull.hiz.as_ref(),
            self.state.planner.view.samples,
        ) else {
            return Ok(0);
        };
        let Some(depth) = hiz.mip_views.first() else {
            return Ok(0);
        };
        // A frame planned for another size than the pyramid holds samples
        // nothing.
        if (depth.width(), depth.height()) != (params.width as usize, params.height as usize) {
            return Ok(0);
        }

        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::ViewSamples);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get view samples compute encoder".into())
                })?,
            ns_string!("view samples"),
        );
        gpu.dispatch(&enc, depth, &params, slot);
        Ok(0)
    }
}

#[cfg(test)]
mod tests;

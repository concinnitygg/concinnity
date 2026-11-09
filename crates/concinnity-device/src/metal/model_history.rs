//! Model-history snapshot compute pass. Once per frame, after the G-buffer
//! pre-pass has read the previous frame's slot, copies this frame's model
//! matrices out of the bindless object buffer into this frame's slot of the
//! model-history ring. Next frame's pre-pass reprojects through what this wrote.
//!
//! The snapshot is a GPU copy rather than a host-built parallel table because
//! the object buffer already carries every model: a host table would write the
//! same 64 bytes per record a second time.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::history::SNAPSHOT_GROUP;
use concinnity_engine::render::shader::programs::abi::history::metal as history;
use concinnity_engine::render::uniforms::ModelHistoryParams;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::dispatch::dispatch_threads;
use super::encode::ComputeEncode;
use super::scoped_encoder::ScopedEncoder;

impl MtlContext {
    // Encode one snapshot dispatch per target slot. `targets` is normally this
    // frame's slot alone; on the frame a rebuild primes the ring it is every
    // slot, so the first pre-pass to read one finds this frame's models rather
    // than an unwritten buffer.
    pub(in crate::metal) fn encode_model_history(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        object_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        targets: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        record_count: usize,
    ) -> RenderResult<()> {
        let Some(pipeline) = &self.gbuffer.history_pipeline else {
            return Ok(());
        };
        if record_count == 0 || targets.is_empty() {
            return Ok(());
        }
        let params = ModelHistoryParams {
            record_count: u32::saturating_from(record_count),
            _pad: [0; 3],
        };
        // No timing attachment: this dispatch is not the `GBufferPrepass`
        // pass, and claiming that pass's slot pair here made both encoders
        // write it, so the reading was one encoder's start against the other's
        // end. The snapshot is a handful of microseconds; the pre-pass it feeds
        // is what the profiler reports.
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get model-history compute encoder".to_string())
                })?,
            ns_string!("model history"),
        );
        enc.set_pipeline(pipeline);
        enc.set_value(&params, history::PARAMS.index);
        enc.set_buffer(object_buffer, 0, history::OBJECTS.index);
        let threads = [u32::saturating_from(record_count), 1, 1];
        for target in targets {
            enc.set_buffer(target, 0, history::HISTORY.index);
            dispatch_threads(&enc, threads, SNAPSHOT_GROUP);
        }
        Ok(())
    }
}

// Build the model-history compute pipeline from the single-source
// `model_history.hlsl` (params buffer(0), objects buffer(1), history
// buffer(2) -- the same slots the encode above binds).
pub(super) fn build_model_history_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    compute_pipeline(device, &super::builtin_shaders::MODEL_HISTORY, hot_reload)
}

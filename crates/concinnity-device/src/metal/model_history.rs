// src/metal/model_history.rs
//
// Model-history snapshot compute pass. Once per frame, after the G-buffer
// pre-pass has read the previous frame's slot, copies this frame's model
// matrices out of the bindless object buffer into this frame's slot of the
// model-history ring. Next frame's pre-pass reprojects through what this wrote.
//
// The snapshot is a GPU copy rather than a host-built parallel table because
// the object buffer already carries every model: a host table would write the
// same 64 bytes per record a second time.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::uniforms::ModelHistoryParams;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder as _, MTLComputePipelineState,
    MTLDevice as _, MTLLibrary as _, MTLSize,
};

use super::context::MtlContext;
use super::encode::ComputeEncode;
use super::pipeline::ns_str;
use super::scoped_encoder::ScopedEncoder;

// Threads per group, matching `[numthreads(64, 1, 1)]` in model_history.slang.
const THREADGROUP: usize = 64;

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
    ) -> Result<(), String> {
        let Some(pipeline) = &self.gbuffer.history_pipeline else {
            return Ok(());
        };
        if record_count == 0 || targets.is_empty() {
            return Ok(());
        }
        let params = ModelHistoryParams {
            record_count: record_count as u32,
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
                .ok_or("failed to get model-history compute encoder")?,
            ns_string!("model history"),
        );
        enc.set_pipeline(pipeline);
        enc.set_value(&params, 0);
        enc.set_buffer(object_buffer, 0, 1);
        let grid = MTLSize {
            width: record_count,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADGROUP,
            height: 1,
            depth: 1,
        };
        for target in targets {
            enc.set_buffer(target, 0, 2);
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        }
        Ok(())
    }
}

// Build the model-history compute pipeline from the single-source
// `model_history.slang` (params buffer(0), objects buffer(1), history
// buffer(2) -- the same slots the encode above binds).
pub(super) fn build_model_history_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let library = super::slang_builtins::MODEL_HISTORY.library(device, hot_reload)?;
    let func = library
        .newFunctionWithName(&ns_str("model_history_kernel"))
        .ok_or("model_history_kernel not found")?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("failed to create model history pipeline: {:?}", e))
}

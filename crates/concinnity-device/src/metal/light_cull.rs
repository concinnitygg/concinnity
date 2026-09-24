//! Clustered binning compute pass. Once per frame, before the Main pass, bins
//! the scene's local lights (the GpuLight buffer bound at fragment buffer(8)) into
//! per-cluster light lists and the reflection probes' influence boxes into
//! per-cluster probe masks, over a screen-tiled, exponential-depth froxel grid.
//! The forward, SSR and transparent passes then shade from only a fragment's
//! cluster's lights and blend only its cluster's probes.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIST_LEN, ClusterParams};
use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _,
    MTLResourceOptions, MTLSize,
};

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::encode::ComputeEncode;
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

// Clustered-lighting GPU state: the binning compute pipeline and the per-cluster
// light-list and probe-mask buffer it writes / the forward pass reads. Both always
// exist (the buffer is bound at fragment buffer(12) even for the brute-force
// fallback); the pipeline runs only on frames with a light or a probe to bin.
pub(crate) struct LightCullState {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub cluster_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
}

impl MtlContext {
    // Encode the clustered binning pass. One thread per cluster; the kernel
    // builds the cluster's world-space AABB and tests each local light's sphere
    // and each probe's influence box against it, writing the surviving indices
    // into `cluster_buffer`. Caller dispatches this before Main, which reads the
    // same buffer.
    pub(in crate::metal) fn encode_light_cull(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        cluster_params: &ClusterParams,
    ) -> RenderResult<u32> {
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::LightCull);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get light-cull compute encoder".to_string())
                })?,
            ns_string!("clustered light cull"),
        );
        enc.set_pipeline(&self.light_cull.pipeline);

        enc.set_value(cluster_params, 0);
        enc.set_buffer(&self.scene.local_light_buffer, 0, 1);
        enc.set_buffer(&self.light_cull.cluster_buffer, 0, 2);
        // This frame's probe records, built before any pass encodes.
        if let Some(records) = self.probe.records_buf.as_deref() {
            enc.set_buffer(records, 0, 3);
        }

        // One thread per cluster; the kernel builds one cluster's list.
        let tg = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: CLUSTER_COUNT as usize,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        Ok(0)
    }
}

// Build the clustered binning compute pipeline from the single-source
// `light_cull.hlsl` (params buffer(0), lights buffer(1), list buffer(2), probe
// records buffer(3) -- the same slots the encode above binds).
pub(super) fn build_light_cull_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    compute_pipeline(device, &super::builtin_shaders::LIGHT_CULL, hot_reload)
}

// Allocate the per-cluster list buffer: CLUSTER_LIST_LEN u32, every cluster's
// light list and then every cluster's probe mask (see `cluster_types.hlsl`).
// Private storage: written only by the compute kernel, read only by shaders.
pub(super) fn build_cluster_light_buffer(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> {
    let len = CLUSTER_LIST_LEN as usize * std::mem::size_of::<u32>();
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| allocation_failed("cluster light buffer"))
}

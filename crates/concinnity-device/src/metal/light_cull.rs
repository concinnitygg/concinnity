// src/metal/light_cull.rs
//
// Clustered light-binning compute pass. Once per frame, before the Main pass,
// bins the scene's local lights (the GpuLight buffer bound at fragment buffer(8))
// into per-cluster index lists over a screen-tiled, exponential-depth froxel
// grid. The forward pass then shades each fragment from only its cluster's
// lights instead of iterating every light.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _,
    MTLLibrary as _, MTLResourceOptions, MTLSize,
};

use crate::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIGHT_LIST_STRIDE, ClusterParams};

use super::context::MtlContext;
use super::pipeline::ns_str;
use super::scoped_encoder::ScopedEncoder;

// Clustered-lighting GPU state: the binning compute pipeline and the per-cluster
// light-index buffer it writes / the forward pass reads. The buffer is always
// allocated (bound at fragment buffer(12) even for the brute-force fallback);
// the pipeline runs only when the world has local lights.
pub(crate) struct LightCullState {
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub cluster_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
}

impl MtlContext {
    // Encode the clustered light-binning pass. One thread per cluster; the kernel
    // builds the cluster's world-space AABB and tests each local light's sphere
    // against it, writing the surviving indices into `cluster_buffer`. Caller
    // dispatches this before Main, which reads the same buffer.
    pub(in crate::metal) fn encode_light_cull(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        cluster_params: &ClusterParams,
    ) -> Result<u32, String> {
        let pipeline = match &self.light_cull.pipeline {
            Some(p) => p,
            None => return Ok(0),
        };

        let desc = objc2_metal::MTLComputePassDescriptor::computePassDescriptor();
        if let Some(t) = &self.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::LightCull);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get light-cull compute encoder")?,
            "clustered light cull",
        );
        enc.setComputePipelineState(pipeline);

        unsafe {
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(cluster_params).cast(),
                std::mem::size_of::<ClusterParams>(),
                0,
            );
            enc.setBuffer_offset_atIndex(Some(&self.local_light_buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.light_cull.cluster_buffer), 0, 2);

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
        }
        Ok(0)
    }
}

// Build the clustered light-binning compute pipeline from the single-source
// `light_cull.slang` (params buffer(0), lights buffer(1), list buffer(2) --
// the same slots the encode above binds).
pub(super) fn build_light_cull_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let library = super::slang_shaders::LIGHT_CULL.library(device, hot_reload)?;
    let func = library
        .newFunctionWithName(&ns_str("light_cull_kernel"))
        .ok_or("light_cull_kernel not found")?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("failed to create light cull pipeline: {:?}", e))
}

// Allocate the per-cluster light-index buffer: CLUSTER_COUNT blocks of
// CLUSTER_LIGHT_LIST_STRIDE u32 (slot 0 = count, slots 1.. = light indices).
// Private storage: written only by the compute kernel, read only by the shader.
pub(super) fn build_cluster_light_buffer(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>, String> {
    let len = (CLUSTER_COUNT * CLUSTER_LIGHT_LIST_STRIDE) as usize * std::mem::size_of::<u32>();
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| "failed to allocate cluster light buffer".to_string())
}

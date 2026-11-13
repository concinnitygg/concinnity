//! Shadow receiver bounds on Metal: the two dispatches that reduce the Hi-Z
//! pyramid's first level into the light-space bounds of what the camera saw in
//! each cascade's band, and the ring of shared buffers the CPU reads them back
//! from. Frame `R` writes slot `R % depth` and `begin` reads that same slot once
//! the frames-in-flight fence has retired the frame that wrote it, so a reading
//! is never waited on and is always exactly `depth` frames old.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::render_types::{
    ReceiverBands, ReceiverBounds, ReceiverBoundsParams, receiver_tiles,
};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::programs::abi::receivers::metal::{merge, reduce};
use concinnity_engine::render::shader::programs::abi::receivers::{MERGE_GROUP, REDUCE_GROUP};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLTexture,
};

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::dispatch::dispatch_groups;
use super::encode::ComputeEncode;
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

/// The reduction's two kernels.
pub(super) struct ReceiverPipelines {
    reduce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    merge: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ReceiverPipelines {
    /// Both kernels, from `shadow_receivers.hlsl`.
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        Ok(Self {
            reduce: compute_pipeline(
                device,
                &super::builtin_shaders::SHADOW_RECEIVERS_REDUCE,
                hot_reload,
            )?,
            merge: compute_pipeline(
                device,
                &super::builtin_shaders::SHADOW_RECEIVERS_MERGE,
                hot_reload,
            )?,
        })
    }
}

/// The reduction's pipelines, its per-tile scratch and the readback ring.
pub(super) struct ShadowReceiversGpu {
    pub(super) pipelines: ReceiverPipelines,
    // One `ReceiverBands` per tile of the depth, GPU only.
    partials: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The tiles `partials` holds room for.
    tiles: u32,
    // One `ReceiverBounds` per frame in flight, shared so the CPU reads it.
    // Created zeroed, so a slot no frame wrote reads back as serial 0.
    outputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl ShadowReceiversGpu {
    /// The reduction for a `size` depth, with a readback slot per frame in
    /// flight.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
        frames_in_flight: usize,
        size: [u32; 2],
    ) -> RenderResult<Self> {
        let outputs = (0..frames_in_flight.max(1))
            .map(|_| new_buffer(device, size_of::<ReceiverBounds>(), true))
            .collect::<RenderResult<Vec<_>>>()?;
        let tiles = tile_count(size);
        Ok(Self {
            pipelines: ReceiverPipelines::build(device, hot_reload)?,
            partials: new_buffer(device, partials_bytes(tiles), false)?,
            tiles,
            outputs,
        })
    }

    /// Room for the tiles of a `size` depth, regrown only when it holds too
    /// few.
    pub(super) fn resize(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: [u32; 2],
    ) -> RenderResult<()> {
        let tiles = tile_count(size);
        if tiles > self.tiles {
            self.partials = new_buffer(device, partials_bytes(tiles), false)?;
            self.tiles = tiles;
        }
        Ok(())
    }

    /// What ring slot `slot` holds, once the fence has retired its writer.
    pub(super) fn read(&self, slot: usize) -> Option<ReceiverBounds> {
        let output = self.outputs.get(slot)?;
        // SAFETY: `output` is shared storage of one `ReceiverBounds`, so `contents()` is a live
        // CPU mapping of it; `begin` waited on the fence that retired the frame writing this
        // slot, and every bit pattern is a valid `ReceiverBounds`.
        Some(unsafe { (output.contents().as_ptr() as *const ReceiverBounds).read() })
    }
}

impl ShadowReceiversGpu {
    // Whether the partials hold every tile of the depth `params` reads.
    fn holds(&self, params: &ReceiverBoundsParams) -> bool {
        tile_count([params.width, params.height]) <= self.tiles
    }

    // Encode both dispatches into `enc`: `depth` reduced by `params` into
    // ring slot `slot`. The caller has checked [`Self::holds`].
    fn dispatch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        depth: &ProtocolObject<dyn MTLTexture>,
        params: &ReceiverBoundsParams,
        slot: usize,
    ) {
        let Some(output) = self.outputs.get(slot) else {
            return;
        };
        let [groups_x, groups_y] = receiver_tiles(params.width, params.height);
        enc.set_pipeline(&self.pipelines.reduce);
        enc.set_texture(depth, reduce::DEPTH.index);
        enc.set_buffer(&self.partials, 0, reduce::PARTIALS.index);
        enc.set_value(params, reduce::PARAMS.index);
        dispatch_groups(enc, [groups_x, groups_y, 1], REDUCE_GROUP);

        // The serial compute encoder orders the merge after every tile.
        enc.set_pipeline(&self.pipelines.merge);
        enc.set_buffer(&self.partials, 0, merge::PARTIALS.index);
        enc.set_buffer(output, 0, merge::BOUNDS.index);
        enc.set_value(params, merge::PARAMS.index);
        dispatch_groups(enc, [1, 1, 1], MERGE_GROUP);
    }
}

fn tile_count([width, height]: [u32; 2]) -> u32 {
    let [x, y] = receiver_tiles(width.max(1), height.max(1));
    x * y
}

fn partials_bytes(tiles: u32) -> usize {
    tiles as usize * size_of::<ReceiverBands>()
}

fn new_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    shared: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let mode = if shared {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };
    device
        .newBufferWithLength_options(bytes.max(16), mode)
        .ok_or_else(|| allocation_failed("shadow receiver buffer"))
}

impl MtlContext {
    // Reduce the Hi-Z pyramid's first level by this frame's receiver
    // parameters into ring slot `slot`. A no-op without the reduction, the
    // pyramid, or a measurement planned this frame.
    pub(in crate::metal) fn encode_shadow_receivers(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        slot: usize,
    ) -> RenderResult<u32> {
        let (Some(gpu), Some(hiz), Some(params)) = (
            self.shadow_receivers.as_ref(),
            self.cull.hiz.as_ref(),
            self.state.planner.shadow.receivers,
        ) else {
            return Ok(0);
        };
        let Some(depth) = hiz.mip_views.first() else {
            return Ok(0);
        };
        // A frame planned for another size than the pyramid or the partials
        // hold measures nothing.
        if (depth.width(), depth.height()) != (params.width as usize, params.height as usize)
            || !gpu.holds(&params)
        {
            return Ok(0);
        }

        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::ShadowReceivers);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get shadow receiver compute encoder".into())
                })?,
            ns_string!("shadow receivers"),
        );
        gpu.dispatch(&enc, depth, &params, slot);
        Ok(0)
    }
}

#[cfg(test)]
mod tests;

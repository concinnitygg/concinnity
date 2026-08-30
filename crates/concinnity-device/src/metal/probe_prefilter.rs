// src/metal/probe_prefilter.rs
//
// The convolution half of a runtime reflection-probe bake: the compute
// pipelines built from `probe_prefilter.slang`, the two cubes one bake works
// between, and the dispatches that turn six captured faces into the prefiltered
// radiance mip chain the specular term samples.
//
// The capture cube is the render target the six faces resolve into, one cube
// slice each, with a mip chain the `probe_downsample` kernel fills. The probe
// cube is the result: mip 0 a firefly-clamped copy of the capture, every mip
// after it a GGX convolution at that mip's roughness. Both are RGBA16Float --
// the faces are captured as halfs, the clamp caps luminance well inside the
// format's range, and it halves what a probe costs in memory against the
// RGBA32Float cube the CPU convolution used to upload.
//
// Nothing here reads back. A dispatch per destination mip, one mip per frame,
// is what replaced the readback plus the off-thread CPU convolution; the whole
// bake now stays on the GPU timeline.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice,
    MTLLibrary as _, MTLPixelFormat, MTLSize, MTLTexture, MTLTextureType, MTLTextureUsage,
};

use concinnity_core::render::reflection_probe::PrefilterPlan;

use super::allocator::{DeviceAllocator, PooledTexture};
use super::descriptors::TextureDesc;
use super::encode::ComputeEncode;
use super::pipeline::ns_str;

// Threadgroup tile size, matching the kernels' `[numthreads(8, 8, 1)]`. The
// third dispatch dimension is the six cube faces, one thread deep.
const PREFILTER_TILE: usize = 8;

// Colour format of both cubes. RGBA16Float is what the faces resolve as, and
// what the read_write views the kernels bind require (an Apple7 device and
// later reads and writes it; the engine's Metal floor is Apple7).
const PROBE_CUBE_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

/// The three compute pipelines a probe bake convolves with, built once from the
/// precompiled `probe_prefilter.slang` variants.
pub(in crate::metal) struct ProbePrefilterPipelines {
    mip0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    downsample: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ggx: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ProbePrefilterPipelines {
    // Build all three kernels. Called from init under the same gate the bake
    // itself needs (the bindless cull pipeline), so a probe never discovers a
    // missing pipeline mid-capture.
    pub(in crate::metal) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> Result<ProbePrefilterPipelines, String> {
        Ok(ProbePrefilterPipelines {
            mip0: build_kernel(
                device,
                &super::slang_shaders::PROBE_MIP0,
                "probe_mip0",
                hot_reload,
            )?,
            downsample: build_kernel(
                device,
                &super::slang_shaders::PROBE_DOWNSAMPLE,
                "probe_downsample",
                hot_reload,
            )?,
            ggx: build_kernel(
                device,
                &super::slang_shaders::PROBE_GGX,
                "probe_ggx",
                hot_reload,
            )?,
        })
    }
}

fn build_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &super::slang_shaders::SlangLib,
    entry: &str,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let library = lib.library(device, hot_reload)?;
    let function = library
        .newFunctionWithName(&ns_str(entry))
        .ok_or_else(|| format!("{entry} not found in its probe prefilter library"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("failed to create {entry} pipeline: {e:?}"))
}

/// The capture cube six faces render into: RGBA16Float, one slice per face, with
/// the mip chain the convolution's source pyramid occupies.
///
/// Created unpooled, like the bake's other render targets: it is transient, and
/// Metal keeps a resource alive while a command buffer references it, so it can
/// be dropped the moment the bake ends without waiting on a fence.
pub(in crate::metal) fn create_capture_cube(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &PrefilterPlan,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = TextureDesc {
        kind: MTLTextureType::TypeCube,
        format: PROBE_CUBE_FORMAT,
        width: plan.face_size() as usize,
        height: plan.face_size() as usize,
        mip_count: plan.mips() as usize,
        usage: MTLTextureUsage(
            MTLTextureUsage::RenderTarget.0
                | MTLTextureUsage::ShaderRead.0
                | MTLTextureUsage::ShaderWrite.0,
        ),
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create capture cube".into())
}

/// The cubes and per-mip views one convolution works between, held by the
/// prefiltering bake slot until the finished probe cube is installed.
pub(in crate::metal) struct PrefilterGpu {
    // The capture, sampled whole (all mips) by the GGX kernel.
    capture: Retained<ProtocolObject<dyn MTLTexture>>,
    // One single-level 2D-array view of the capture per mip: mip M is the
    // downsample's source and mip M+1 its destination, so no dispatch reads the
    // texels it writes.
    capture_mip_views: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // The prefiltered radiance cube this bake produces.
    probe: PooledTexture,
    // One single-level 2D-array view of the probe cube per mip, the destination
    // of the mip-0 copy and of each GGX dispatch.
    probe_mip_views: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
}

impl PrefilterGpu {
    /// Take ownership of a finished `capture` and allocate the probe cube it
    /// convolves into, plus the per-mip write views both need.
    pub(in crate::metal) fn new(
        alloc: &DeviceAllocator,
        capture: Retained<ProtocolObject<dyn MTLTexture>>,
        plan: &PrefilterPlan,
    ) -> Result<PrefilterGpu, String> {
        let desc = TextureDesc {
            kind: MTLTextureType::TypeCube,
            format: PROBE_CUBE_FORMAT,
            width: plan.face_size() as usize,
            height: plan.face_size() as usize,
            mip_count: plan.mips() as usize,
            usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0),
            ..Default::default()
        }
        .build();
        let probe = alloc.alloc_texture(&desc)?;
        let capture_mip_views = mip_array_views(&capture, plan.mips(), "capture")?;
        let probe_mip_views = mip_array_views(&probe, plan.mips(), "probe")?;
        Ok(PrefilterGpu {
            capture,
            capture_mip_views,
            probe,
            probe_mip_views,
        })
    }

    /// The finished cube, handed to the probe pool at install.
    pub(in crate::metal) fn into_probe_cube(self) -> PooledTexture {
        self.probe
    }
}

// One single-level 2D-array view per mip of a cube texture. A cube is a
// six-slice array, so the view is what lets a kernel address (x, y, face)
// directly; the format is the parent's, so no reinterpretation occurs.
fn mip_array_views(
    texture: &ProtocolObject<dyn MTLTexture>,
    mips: u32,
    label: &str,
) -> Result<Vec<Retained<ProtocolObject<dyn MTLTexture>>>, String> {
    (0..mips)
        .map(|mip| {
            // SAFETY: `mip` is in `0..mips` and the texture was created with
            // `mips` levels and the six slices of a cube; the view shares the
            // parent's pixel format, so it reinterprets nothing.
            unsafe {
                texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                    PROBE_CUBE_FORMAT,
                    MTLTextureType::Type2DArray,
                    NSRange::new(mip as usize, 1),
                    NSRange::new(0, 6),
                )
            }
            .ok_or_else(|| format!("probe: failed to create {label} mip {mip} view"))
        })
        .collect()
}

impl super::context::MtlContext {
    /// Encode the source pyramid: the firefly-clamped copy of the capture into
    /// probe-cube mip 0, then the box reduction of each capture mip into the
    /// next. Both run in one encoder because Metal orders successive dispatches
    /// in a serial compute encoder, which is exactly the chain's dependency.
    ///
    /// These are the cheap dispatches (a few taps per texel), so they all go in
    /// the frame that starts the convolution; the GGX mips that follow are the
    /// expensive ones and take a frame each.
    pub(in crate::metal) fn encode_probe_pyramid(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        let enc = super::scoped_encoder::ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoder()
                .ok_or("probe: failed to get prefilter compute encoder")?,
            "probe-pyramid",
        );

        let params = plan.mip0_params();
        enc.set_pipeline(&pipelines.mip0);
        enc.set_value(&params, 0);
        enc.set_texture(gpu.capture_mip_views[0].as_ref(), 0);
        enc.set_texture(gpu.probe_mip_views[0].as_ref(), 1);
        dispatch_cube(&enc, plan.face_size());

        for mip in 1..plan.mips() {
            let params = plan.downsample_params(mip);
            enc.set_pipeline(&pipelines.downsample);
            enc.set_value(&params, 0);
            enc.set_texture(gpu.capture_mip_views[(mip - 1) as usize].as_ref(), 0);
            enc.set_texture(gpu.capture_mip_views[mip as usize].as_ref(), 1);
            dispatch_cube(&enc, plan.mip_face_size(mip));
        }
        Ok(())
    }

    /// Encode the GGX convolution producing probe-cube mip `dst_mip`, sampling
    /// the whole capture pyramid through the engine's cube sampler (linear,
    /// clamped, mipmapped -- the solid-angle lod the kernel picks needs the
    /// trilinear tap).
    pub(in crate::metal) fn encode_probe_ggx_mip(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
        dst_mip: u32,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        let enc = super::scoped_encoder::ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoder()
                .ok_or("probe: failed to get prefilter compute encoder")?,
            "probe-ggx",
        );
        let params = plan.ggx_params(dst_mip);
        enc.set_pipeline(&pipelines.ggx);
        enc.set_value(&params, 0);
        enc.set_texture(gpu.capture.as_ref(), 0);
        enc.set_sampler(&self.cube_sampler, 0);
        enc.set_texture(gpu.probe_mip_views[dst_mip as usize].as_ref(), 1);
        dispatch_cube(&enc, plan.mip_face_size(dst_mip));
        Ok(())
    }
}

// Dispatch one thread per texel of a `size`-square cube face, six faces deep.
// The kernels bounds-guard against `dst_size`, so a non-uniform remainder
// returns early.
fn dispatch_cube(enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, size: u32) {
    let grid = MTLSize {
        width: size.max(1) as usize,
        height: size.max(1) as usize,
        depth: 6,
    };
    let tg = MTLSize {
        width: PREFILTER_TILE,
        height: PREFILTER_TILE,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

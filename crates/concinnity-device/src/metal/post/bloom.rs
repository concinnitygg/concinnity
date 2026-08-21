// src/metal/post/bloom.rs
//
// Bloom pass: prefilter + downsample chain + additive upsample. Pipelines,
// mip-chain target allocation, and per-frame encoder live together so the
// effect is a single unit Vulkan / DirectX can mirror.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState, MTLTexture, MTLTextureUsage,
};

use crate::metal::context::MtlContext;
use crate::metal::descriptors::TextureDesc;
use crate::metal::post::fullscreen::{
    FullscreenBlend, FullscreenPass, PassTimer, build_slang_fullscreen_pipeline,
};
use crate::metal::slang_shaders::{BLOOM_DOWNSAMPLE, BLOOM_PREFILTER, BLOOM_UPSAMPLE, SlangLib};

// Pixel format of every mip in the bloom chain, including the `bloom_top` mip
// the transient pool backs.
pub(crate) const BLOOM_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

// Pipelines

// The three fullscreen-triangle pipelines that make up the bloom chain. All
// target single-sample `BLOOM_FORMAT` bloom mips; `upsample` blends additively
// so each upsample pass accumulates onto the downsampled content already in
// the destination mip.
pub(crate) struct BloomPipelines {
    // HDR resolve -> mip 0: soft-knee threshold + Karis 13-tap downsample.
    pub prefilter: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // mip i-1 -> mip i: 13-tap downsample.
    pub downsample: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // mip i+1 -> mip i: 9-tap tent upsample, additively blended.
    pub upsample: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

// Build the bloom prefilter / downsample / upsample pipelines from the
// single-source `bloom.slang`. The filter kernels are the Jimenez "Next
// Generation Post Processing in Call of Duty" 13-tap downsample + 9-tap tent
// upsample; the first downsample applies a Karis luma-weighted average to
// suppress fireflies and a soft-knee luminance threshold.
pub(crate) fn build_bloom_pipelines(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<BloomPipelines, String> {
    let build = |lib: &SlangLib, blend: FullscreenBlend| {
        build_slang_fullscreen_pipeline(device, lib, BLOOM_FORMAT, blend, hot_reload)
    };

    Ok(BloomPipelines {
        prefilter: build(&BLOOM_PREFILTER, FullscreenBlend::Replace)?,
        downsample: build(&BLOOM_DOWNSAMPLE, FullscreenBlend::Replace)?,
        // One/One additive: each upsampled mip is added onto the destination
        // mip's existing downsampled content.
        upsample: build(&BLOOM_UPSAMPLE, FullscreenBlend::Additive)?,
    })
}

// Targets

// Off-screen bloom mip chain. `mips[0]` is half the HDR resolve resolution;
// each subsequent mip halves again. The prefilter + downsample passes fill
// `mips[0..N]`, the additive upsample passes accumulate back down to
// `mips[0]`, and the composite pass samples `mips[0]`. All mips are
// single-sample `BLOOM_FORMAT`, `ShaderRead | RenderTarget`, GPU-private.
//
// `mips[0]` is the graph's `bloom_top` transient and belongs to the transient
// pool, which may back it with memory `ao_output` also uses; the chain only
// borrows the handle. Every mip below it is committed and owned here.
pub(crate) struct BloomTargets {
    // One texture per mip level, largest first. Always non-empty.
    pub mips: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // HDR resolve resolution the chain was built for; `mips[0]` is half this.
    pub width: u32,
    pub height: u32,
}

// Number of mip levels in the bloom chain for a given HDR resolve resolution.
// Clamped to 4..=6 -- enough octaves for a wide, soft glow without spending
// a dozen render passes on sub-pixel mips.
fn bloom_mip_count(width: u32, height: u32) -> u32 {
    let min_dim = width.min(height).max(1);
    // mips[0] is already half-res, so subtract one octave before clamping.
    let levels = (min_dim as f32).log2().floor() as i32 - 1;
    levels.clamp(4, 6) as u32
}

// Create the bloom mip chain for an HDR resolve target of `width`x`height`.
// `mips[i]` has resolution `(width >> (i + 1), height >> (i + 1))`, floored
// at one texel. `bloom_top` is the pool's `mips[0]`, which the pool sizes from
// the graph's own half-drawable desc.
pub(crate) fn create_bloom_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    bloom_top: Retained<ProtocolObject<dyn MTLTexture>>,
) -> Result<BloomTargets, String> {
    let full_w = width.max(1);
    let full_h = height.max(1);
    let count = bloom_mip_count(full_w, full_h);

    let mut mips = Vec::with_capacity(count as usize);
    mips.push(bloom_top);
    for i in 1..count {
        let mw = (full_w >> (i + 1)).max(1) as usize;
        let mh = (full_h >> (i + 1)).max(1) as usize;
        let desc = TextureDesc {
            format: BLOOM_FORMAT,
            width: mw,
            height: mh,
            usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
            ..Default::default()
        }
        .build();
        let tex = device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to create bloom mip {} texture", i))?;
        mips.push(tex);
    }

    Ok(BloomTargets {
        mips,
        width: full_w,
        height: full_h,
    })
}

// Encoder

impl MtlContext {
    // Encode the bloom prefilter, downsample, and additive upsample passes.
    //
    // Runs between the TAA resolve and the composite pass. Each pass is one
    // fullscreen-triangle draw into a bloom mip; Metal inserts the texture
    // read/write hazards between passes automatically. On return `mips[0]`
    // holds the accumulated bloom that the composite pass samples.
    // `scene_color` is the post-TAA scene colour (or `hdr_resolve` when TAA
    // is off) that the prefilter pass thresholds.
    pub(in crate::metal) fn encode_bloom(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_color: &ProtocolObject<dyn objc2_metal::MTLTexture>,
    ) -> Result<u32, String> {
        // Scene-less worlds build no bloom pipelines and the graph never
        // inserts the Bloom pass, so this is a defensive no-op there.
        let Some(bloom_pipelines) = &self.bloom_pipelines else {
            return Ok(0);
        };
        let mips = &self.bloom_targets.mips;
        let n = mips.len();

        // Prefilter: hdr_resolve -> mips[0] (soft-knee threshold + Karis 13-tap).
        // Bloom's GPU-timing span runs from this prefilter through the final
        // upsample: mark the start sample here (and the end on the last upsample
        // below). With a single mip (no downsample / upsample) the prefilter is
        // the only encoder and owns both samples.
        let prefilter_timer = if n <= 1 {
            PassTimer::Whole(crate::metal::pass_timing::PassId::Bloom)
        } else {
            PassTimer::First(crate::metal::pass_timing::PassId::Bloom)
        };
        self.fullscreen_pass(
            cmd_buf,
            FullscreenPass {
                target: mips[0].as_ref(),
                load: MTLLoadAction::DontCare,
                timer: prefilter_timer,
                pipeline: &bloom_pipelines.prefilter,
                label: "bloom prefilter",
            },
            // SAFETY: each `setBytes` pointer is derived from a live borrow with that type's
            // `size_of` as the length, and every bound resource outlives the encoder; the indices
            // are the slots the shaders declare.
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(scene_color), 0);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&self.post_process).cast(),
                    std::mem::size_of::<crate::gfx::render_types::PostProcessParams>(),
                    0,
                );
            },
        )?;

        // Downsample chain: mips[i-1] -> mips[i].
        for i in 1..n {
            self.fullscreen_pass(
                cmd_buf,
                FullscreenPass {
                    target: mips[i].as_ref(),
                    load: MTLLoadAction::DontCare,
                    timer: PassTimer::None,
                    pipeline: &bloom_pipelines.downsample,
                    label: "bloom downsample",
                },
                // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at
                // the buffer/texture indices the shaders declare.
                |enc| unsafe {
                    enc.setFragmentTexture_atIndex(Some(mips[i - 1].as_ref()), 0);
                    enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                },
            )?;
        }

        // Upsample chain: mips[i+1] -> mips[i], additively blended onto the
        // downsampled content already in mips[i]. Walks back down to mips[0].
        // The `i == 0` iteration is the chain's final encoder, so it records the
        // end timing sample; the intermediate upsamples write none.
        for i in (0..n - 1).rev() {
            let timer = if i == 0 {
                PassTimer::Last(crate::metal::pass_timing::PassId::Bloom)
            } else {
                PassTimer::None
            };
            self.fullscreen_pass(
                cmd_buf,
                FullscreenPass {
                    target: mips[i].as_ref(),
                    load: MTLLoadAction::Load,
                    timer,
                    pipeline: &bloom_pipelines.upsample,
                    label: "bloom upsample",
                },
                // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at
                // the buffer/texture indices the shaders declare.
                |enc| unsafe {
                    enc.setFragmentTexture_atIndex(Some(mips[i + 1].as_ref()), 0);
                    enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                },
            )?;
        }

        Ok(0)
    }
}

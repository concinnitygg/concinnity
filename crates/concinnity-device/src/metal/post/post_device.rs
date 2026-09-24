//! Metal's implementation of the shared fullscreen post-pass seam
//! (`render::post::device::PostPassDevice`). Thin, because Metal's own API is
//! already close to the seam's shape: a pipeline is a program plus an attachment
//! format plus a blend, a target is a texture descriptor, and a draw is one
//! render encoder with slot-indexed fragment binds. Everything here is a
//! translation, not a mechanism.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, PostTiming,
    resolved_texture,
};
use concinnity_core::render::post::program::{PostProgram, PostProgramBindings};
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLDevice as _, MTLLoadAction, MTLRenderPipelineState, MTLSamplerState,
    MTLTexture,
};

use crate::metal::encode::RenderEncode;
use crate::metal::error::allocation_failed;
use crate::metal::pass_timing::PassTimingResources;
use crate::metal::post::fullscreen::{
    FullscreenBlend, FullscreenPass, PassTimer, build_fullscreen_pipeline, encode_fullscreen_pass,
};
use crate::metal::probe_set::{ProbeBindings, ProbeSlots};
use crate::metal::transient_pool::{pixel_format, texture_descriptor_for};

// Buffer slots a probe-reading post program declares its `ProbeSet`, the
// cluster params, its probe records and the cluster lists at, after the
// constants at buffer(0): the registers ssr.hlsl names. The cube array takes
// the texture slot after the declared sources.
fn probe_slots(sources: usize) -> ProbeSlots {
    ProbeSlots {
        set: 1,
        records: 5,
        cubes: Some(sources),
        cluster: Some((2, 6)),
    }
}

// A built fullscreen post pipeline plus what its program declares, so a draw
// can check what it was handed.
pub(crate) struct MtlPostPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    bindings: PostProgramBindings,
}

// The Metal handles a shared post pass builds and encodes through. Borrowed
// rather than owned so it can be assembled at init (where only the device and
// the samplers exist yet) and again per frame off the live context.
pub(in crate::metal) struct MtlPostDevice<'a> {
    pub device: &'a ProtocolObject<dyn objc2_metal::MTLDevice>,
    // The linear clamp-to-edge state every screen-space source is read through.
    pub sampler: &'a ProtocolObject<dyn MTLSamplerState>,
    // The trilinear clamp-to-edge state an environment cube is read through.
    pub cube_sampler: &'a ProtocolObject<dyn MTLSamplerState>,
    // The probe set a probe-reading program binds. Absent at init, where no
    // draw is encoded.
    pub probes: Option<ProbeBindings<'a>>,
    // GPU-timing resources, absent when timing is off.
    pub timing: Option<&'a PassTimingResources>,
    pub hot_reload: bool,
}

fn blend(blend: PostBlend) -> FullscreenBlend {
    match blend {
        PostBlend::Replace => FullscreenBlend::Replace,
        PostBlend::Additive => FullscreenBlend::Additive,
        PostBlend::PremultipliedOver => FullscreenBlend::PremultipliedOver,
    }
}

fn load_action(load: PostLoadOp) -> MTLLoadAction {
    match load {
        PostLoadOp::DontCare => MTLLoadAction::DontCare,
        PostLoadOp::Load => MTLLoadAction::Load,
    }
}

fn timer(timing: PostTiming) -> PassTimer {
    match timing {
        PostTiming::None => PassTimer::None,
        PostTiming::Whole(id) => PassTimer::Whole(id),
        PostTiming::First(id) => PassTimer::First(id),
        PostTiming::Last(id) => PassTimer::Last(id),
    }
}

impl MtlPostDevice<'_> {
    fn sampler_for(&self, sampler: PostSampler) -> &ProtocolObject<dyn MTLSamplerState> {
        match sampler {
            PostSampler::LinearClamp => self.sampler,
            PostSampler::LinearCube => self.cube_sampler,
        }
    }
}

impl PostPassDevice for MtlPostDevice<'_> {
    type Recorder = ProtocolObject<dyn MTLCommandBuffer>;
    type Pipeline = MtlPostPipeline;
    type Target = Retained<ProtocolObject<dyn MTLTexture>>;
    type TextureRef<'a> = &'a ProtocolObject<dyn MTLTexture>;
    type Attachment<'a> = &'a ProtocolObject<dyn MTLTexture>;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend_mode: PostBlend,
    ) -> RenderResult<Self::Pipeline> {
        let state = build_fullscreen_pipeline(
            self.device,
            program.program(),
            pixel_format(format),
            blend(blend_mode),
            self.hot_reload,
        )?;
        Ok(MtlPostPipeline {
            state,
            bindings: program.bindings(),
        })
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> RenderResult<Self::Target> {
        let spec = resolved_texture(label, desc, extent);
        let desc = texture_descriptor_for(&spec);
        self.device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| allocation_failed(format_args!("the {label} post target")))
    }

    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a> {
        target.as_ref()
    }

    fn target_attachment<'a>(&self, target: &'a Self::Target) -> Self::Attachment<'a> {
        target.as_ref()
    }

    fn encode(&self, rec: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> RenderResult<()> {
        let bindings = draw.pipeline.bindings;
        draw.check(bindings)?;
        let probes = match (bindings.probes, self.probes) {
            (false, _) => None,
            (true, Some(probes)) => Some(probes),
            (true, None) => {
                return Err(RenderError::Other(format!(
                    "{}: the program reads the reflection-probe set, but this device holds none",
                    draw.label
                )));
            }
        };
        encode_fullscreen_pass(
            rec,
            self.timing,
            FullscreenPass {
                target: draw.target,
                load: load_action(draw.load),
                timer: timer(draw.timing),
                pipeline: &draw.pipeline.state,
                label: draw.label,
            },
            |enc| {
                for (slot, bind) in draw.binds.iter().enumerate() {
                    enc.set_fragment_texture(bind.texture, slot);
                    // Every post source occupies a texture and a sampler at
                    // the same index, so a source's sampler slot is its texture
                    // slot.
                    enc.set_fragment_sampler(self.sampler_for(bind.sampler), slot);
                }
                if !draw.constants.is_empty() {
                    enc.set_fragment_bytes(draw.constants, 0);
                }
                if let Some(probes) = probes {
                    // The cube array's sampler takes the sampler slot after the
                    // declared sources, beside the array itself.
                    enc.set_fragment_sampler(self.cube_sampler, draw.binds.len());
                    probes.bind(enc, probe_slots(draw.binds.len()));
                }
            },
        )?;
        Ok(())
    }
}

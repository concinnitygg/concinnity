// src/metal/post/post_device.rs
//
// Metal's implementation of the shared fullscreen post-pass seam
// (`gfx::post::PostPassDevice`). Thin, because Metal's own API is already close
// to the seam's shape: a pipeline is a program plus an attachment format plus a
// blend, a target is a texture descriptor, and a draw is one render encoder with
// slot-indexed fragment binds. Everything here is a translation, not a
// mechanism.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, resolved_texture,
};
use concinnity_core::render::post::program::PostProgram;
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLDevice as _, MTLLoadAction, MTLRenderPipelineState, MTLSamplerState,
    MTLTexture,
};

use crate::metal::encode::RenderEncode;
use crate::metal::pass_timing::PassTimingResources;
use crate::metal::post::fullscreen::{
    FullscreenBlend, FullscreenPass, PassTimer, build_slang_fullscreen_pipeline,
    encode_fullscreen_pass,
};
use crate::metal::slang_builtins::{SlangLib, TAA_FRAG};
use crate::metal::transient_pool::{pixel_format, texture_descriptor_for};

// The Metal handles a shared post pass builds and encodes through. Borrowed
// rather than owned so it can be assembled at init (where only the device and
// the sampler exist yet) and again per frame off the live context.
pub(in crate::metal) struct MtlPostDevice<'a> {
    pub device: &'a ProtocolObject<dyn objc2_metal::MTLDevice>,
    // The linear clamp-to-edge state every screen-space source is read through.
    pub sampler: &'a ProtocolObject<dyn MTLSamplerState>,
    // GPU-timing resources, absent when timing is off.
    pub timing: Option<&'a PassTimingResources>,
    pub hot_reload: bool,
}

// The Metal library a post program's fragment comes from.
fn library(program: PostProgram) -> &'static SlangLib {
    match program {
        PostProgram::TaaResolve => &TAA_FRAG,
    }
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

impl PostPassDevice for MtlPostDevice<'_> {
    type Recorder = ProtocolObject<dyn MTLCommandBuffer>;
    type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
    type Target = Retained<ProtocolObject<dyn MTLTexture>>;
    type TextureRef<'a> = &'a ProtocolObject<dyn MTLTexture>;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend_mode: PostBlend,
    ) -> Result<Self::Pipeline, String> {
        build_slang_fullscreen_pipeline(
            self.device,
            library(program),
            pixel_format(format),
            blend(blend_mode),
            self.hot_reload,
        )
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> Result<Self::Target, String> {
        let spec = resolved_texture(label, desc, extent);
        let desc = texture_descriptor_for(&spec);
        self.device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to create the {label} post target"))
    }

    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a> {
        target.as_ref()
    }

    fn encode(&self, rec: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> Result<(), String> {
        encode_fullscreen_pass(
            rec,
            self.timing,
            FullscreenPass {
                target: draw.target.as_ref(),
                load: load_action(draw.load),
                timer: match draw.timing {
                    Some(id) => PassTimer::Whole(id),
                    None => PassTimer::None,
                },
                pipeline: draw.pipeline,
                label: draw.label,
            },
            |enc| {
                for (slot, bind) in draw.binds.iter().enumerate() {
                    enc.set_fragment_texture(bind.texture, slot);
                    // slangc lowers each combined `Sampler2D` to a texture and a
                    // sampler at the same index, so a source's sampler slot is
                    // its texture slot.
                    match bind.sampler {
                        PostSampler::LinearClamp => enc.set_fragment_sampler(self.sampler, slot),
                    }
                }
                if !draw.constants.is_empty() {
                    enc.set_fragment_bytes(draw.constants, 0);
                }
            },
        )
    }
}

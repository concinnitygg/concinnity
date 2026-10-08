// The reactive mask's attachment on the passes that write it: the target
// itself lives with the HDR set, and `core::render::reactive_mask` decides how
// each writer treats it this frame.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::reactive_mask::ReactiveWrite;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLCommandBuffer, MTLLoadAction,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLStoreAction,
};

use super::context::MtlContext;
use super::scoped_encoder::ScopedEncoder;
use super::texture::REACTIVE_MASK_FORMAT;

// The color slot a writer's pipeline and pass put the mask at.
pub(super) const REACTIVE_MASK_TARGET: usize = 1;

// Declare the mask's target on a writer pipeline: max-blended, so the most
// reactive layer over a pixel wins whatever order the layers draw in.
pub(super) fn declare_target(desc: &MTLRenderPipelineDescriptor) {
    // SAFETY: plain descriptor property setters on a color slot the pipeline
    // declares.
    unsafe {
        let ca = desc
            .colorAttachments()
            .objectAtIndexedSubscript(REACTIVE_MASK_TARGET);
        ca.setPixelFormat(REACTIVE_MASK_FORMAT);
        ca.setBlendingEnabled(true);
        ca.setRgbBlendOperation(MTLBlendOperation::Max);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::One);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::One);
        ca.setAlphaBlendOperation(MTLBlendOperation::Max);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::One);
    }
}

impl MtlContext {
    // Attach the mask to a writer's pass as `write` says. An unstored mask is
    // still attached, since the pipeline declares it, but never reaches memory.
    pub(super) fn attach_reactive_mask(
        &self,
        desc: &MTLRenderPassDescriptor,
        write: ReactiveWrite,
    ) {
        let (load, store) = match write {
            ReactiveWrite::Unstored => (MTLLoadAction::DontCare, MTLStoreAction::DontCare),
            ReactiveWrite::Clear => (MTLLoadAction::Clear, MTLStoreAction::Store),
            ReactiveWrite::Load => (MTLLoadAction::Load, MTLStoreAction::Store),
        };
        // SAFETY: plain descriptor property setters on a color slot the writer's
        // pipeline declares; the texture is owned by `self`.
        unsafe {
            let ca = desc
                .colorAttachments()
                .objectAtIndexedSubscript(REACTIVE_MASK_TARGET);
            ca.setTexture(Some(self.targets.hdr.reactive_mask.as_ref()));
            ca.setLoadAction(load);
            ca.setStoreAction(store);
            ca.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
        }
    }

    // Clear the mask for a writer that draws nothing this frame, so a reader
    // never sees an earlier frame's mask. A no-op unless `write` clears.
    pub(super) fn clear_reactive_mask(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        write: ReactiveWrite,
    ) -> RenderResult<()> {
        if write != ReactiveWrite::Clear {
            return Ok(());
        }
        let desc = MTLRenderPassDescriptor::new();
        self.attach_reactive_mask(&desc, write);
        let _enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get reactive mask clear encoder".into())
                })?,
            ns_string!("reactive mask: clear"),
        );
        Ok(())
    }
}

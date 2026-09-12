//! A recording [`PostPassDevice`] for the shared passes' unit tests.
//!
//! It builds nothing: a pipeline is the program and blend it was asked for, a
//! target is an index into the list of targets it was asked to create, and a
//! draw is checked against its program's declaration and then recorded, so a
//! test can assert what a pass drew, into what, through which binds.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::render::render_graph::{PixelFormat, TextureDesc};

use super::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, PostTargetState,
    PostTiming, resolve_extent,
};
use super::program::PostProgram;

/// A texture or attachment the mock names: one of its own targets, or a value
/// a test supplied as some other subsystem's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockTexture {
    Target(usize),
    External(u32),
}

/// A built mock pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MockPipeline {
    pub program: PostProgram,
    pub format: PixelFormat,
    pub blend: PostBlend,
}

/// One recorded draw.
#[derive(Clone, Debug)]
pub(crate) struct MockDraw {
    pub program: PostProgram,
    pub target: MockTexture,
    pub state: PostTargetState,
    pub load: PostLoadOp,
    pub timing: PostTiming,
    pub binds: Vec<(MockTexture, PostSampler)>,
    pub constants: Vec<u8>,
    pub label: String,
}

/// One created target: its label and the extent it resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MockTarget {
    pub label: &'static str,
    pub extent: PostExtent,
}

#[derive(Default)]
pub(crate) struct MockDevice {
    pub targets: RefCell<Vec<MockTarget>>,
    pub draws: RefCell<Vec<MockDraw>>,
}

impl MockDevice {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl PostPassDevice for MockDevice {
    type Recorder = ();
    type Pipeline = MockPipeline;
    type Target = usize;
    type TextureRef<'a> = MockTexture;
    type Attachment<'a> = MockTexture;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> Result<MockPipeline, String> {
        Ok(MockPipeline {
            program,
            format,
            blend,
        })
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> Result<usize, String> {
        let mut targets = self.targets.borrow_mut();
        targets.push(MockTarget {
            label,
            extent: resolve_extent(desc, extent),
        });
        Ok(targets.len() - 1)
    }

    fn target_ref(&self, target: &usize) -> MockTexture {
        MockTexture::Target(*target)
    }

    fn target_attachment(&self, target: &usize) -> MockTexture {
        MockTexture::Target(*target)
    }

    fn encode(&self, _rec: &(), draw: &PostDraw<'_, '_, Self>) -> Result<(), String> {
        draw.check(draw.pipeline.program.bindings())?;
        self.draws.borrow_mut().push(MockDraw {
            program: draw.pipeline.program,
            target: draw.target,
            state: draw.state,
            load: draw.load,
            timing: draw.timing,
            binds: draw.binds.iter().map(|b| (b.texture, b.sampler)).collect(),
            constants: draw.constants.to_vec(),
            label: draw.label.to_string(),
        });
        Ok(())
    }
}

//! A recording [`PostPassDevice`] for the shared passes' unit tests.
//!
//! It builds nothing: a pipeline is the program and blend it was asked for, a
//! target is an index into the list of targets it was asked to create, and a
//! draw is checked against its program's declaration and then recorded, so a
//! test can assert what a pass drew, into what, through which binds.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use crate::render::error::{RenderError, RenderResult};
use crate::render::render_graph::{PixelFormat, TextureDesc};

use super::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, PostTargetState,
    PostTiming, check_level, resolve_extent,
};
use super::program::PostProgram;

/// A texture or attachment the mock names: one of its own targets, or a value
/// a test supplied as some other subsystem's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockTexture {
    Target(usize),
    /// One mip level of one of its own targets.
    Level(usize, u32),
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
    pub levels: u32,
}

#[derive(Default)]
pub(crate) struct MockDevice {
    pub targets: RefCell<Vec<MockTarget>>,
    pub draws: RefCell<Vec<MockDraw>>,
    // How many more targets may be created before creation fails.
    create_budget: Cell<Option<usize>>,
}

impl MockDevice {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Let `n` more targets be created, then fail every creation after them.
    pub(crate) fn fail_creates_after(&self, n: usize) {
        self.create_budget.set(Some(n));
    }

    fn level(&self, target: usize, level: u32) -> RenderResult<MockTexture> {
        let t = &self.targets.borrow()[target];
        check_level(t.label, level, t.levels)?;
        Ok(MockTexture::Level(target, level))
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
    ) -> RenderResult<MockPipeline> {
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
    ) -> RenderResult<usize> {
        if let Some(budget) = self.create_budget.get() {
            let Some(left) = budget.checked_sub(1) else {
                return Err(RenderError::Other(format!("{label}: creation failed")));
            };
            self.create_budget.set(Some(left));
        }
        let mut targets = self.targets.borrow_mut();
        targets.push(MockTarget {
            label,
            extent: resolve_extent(desc, extent),
            levels: desc.mip_levels.max(1),
        });
        Ok(targets.len() - 1)
    }

    fn target_ref(&self, target: &usize) -> MockTexture {
        MockTexture::Target(*target)
    }

    fn target_attachment(&self, target: &usize) -> MockTexture {
        MockTexture::Target(*target)
    }

    fn target_level_ref(&self, target: &usize, level: u32) -> RenderResult<MockTexture> {
        self.level(*target, level)
    }

    fn target_level_attachment(&self, target: &usize, level: u32) -> RenderResult<MockTexture> {
        self.level(*target, level)
    }

    fn encode(&self, _rec: &(), draw: &PostDraw<'_, '_, Self>) -> RenderResult<()> {
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

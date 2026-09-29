//! Building a world's replacement pipelines away from the frame thread.
//!
//! Creating a pipeline is where a driver turns shader code into GPU machine
//! code, which costs tens to hundreds of milliseconds for a world Shader. A
//! [`PipelineBuilder`] does that on whichever thread calls it, against the live
//! device, and hands back [`PreparedPipelines`] for
//! [`LiveEdit::update_world_shader`](super::LiveEdit::update_world_shader) or
//! [`LiveEdit::replace_sdf_volume_pipelines`](super::LiveEdit::replace_sdf_volume_pipelines)
//! to swap in without building.

use alloc::boxed::Box;
use core::any::Any;
use core::fmt;

use crate::components::ShaderPrograms;
use crate::components::sdf_programs::SdfPrograms;
use crate::render::error::RenderResult;
use crate::render::shader_programs::raymarch::VolumeFlags;

/// Pipelines a [`PipelineBuilder`] built, opaque to everything but the backend
/// that built them.
pub struct PreparedPipelines(Box<dyn Any + Send>);

impl PreparedPipelines {
    /// Wrap a backend's own pipeline set.
    pub fn new<T: Any + Send>(pipelines: T) -> Self {
        Self(Box::new(pipelines))
    }

    /// The pipeline set as `T`, or `None` when it holds another type.
    pub fn downcast<T: Any>(self) -> Option<T> {
        self.0.downcast::<T>().ok().map(|pipelines| *pipelines)
    }
}

impl fmt::Debug for PreparedPipelines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PreparedPipelines(..)")
    }
}

/// Builds world Shader and SdfVolume pipelines against a live backend's device,
/// from any thread.
///
/// A builder captures what pipeline creation reads when
/// [`LiveEdit::pipeline_builder`](super::LiveEdit::pipeline_builder) hands it
/// out. The backend checks what it built against at swap time, and builds the
/// pipeline itself when the two no longer match.
pub trait PipelineBuilder: Send + Sync {
    /// Build shader bucket `bucket`'s main-pass pipeline from `programs`.
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines>;

    /// Build every pipeline a volume with `flags` draws with from `programs`.
    /// `label` names the volume in an error.
    fn sdf_volume(
        &self,
        programs: &SdfPrograms,
        flags: VolumeFlags,
        label: &str,
    ) -> RenderResult<PreparedPipelines>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_pipelines_downcast_only_to_the_type_they_hold() {
        assert_eq!(PreparedPipelines::new(7u32).downcast::<u32>(), Some(7));
        assert_eq!(PreparedPipelines::new(7u32).downcast::<u64>(), None);
    }
}

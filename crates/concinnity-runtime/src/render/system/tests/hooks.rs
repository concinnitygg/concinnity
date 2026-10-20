//! The `GraphicsSystem` injection hooks for unit tests: a fixed GPU profile and a
//! factory that builds a core `RecordingBackend` in place of the real
//! `init_backend`. A test's settings store is the `StoredSettings` resource it
//! inserts into the world.

use concinnity_core::render::backend::error::RenderResult;
use concinnity_core::render::backend::init::BackendInit;
use concinnity_core::render::backend::recording::{RecordingBackend, SharedRecording};
use concinnity_core::render::backend::{GpuProfile, RenderBackend};
use std::sync::Arc;

// Everything a test injects into GraphicsSystem before init: the GPU profile
// (so no device handle is created) and the backend factory run in place of
// the real `init_backend`.
pub(crate) struct TestHooks {
    pub gpu_profile: GpuProfile,
    pub(crate) backend_factory: BackendFactory,
}

pub(crate) type BackendFactory =
    Box<dyn FnMut(BackendInit<'_>) -> RenderResult<Box<dyn RenderBackend>> + Send>;

// Hooks with an unclassified GPU: the preset resolves to Auto with no ceiling,
// so the world's authored values pass through unclamped. Returns the shared
// state for assertions.
pub(crate) fn recording_hooks() -> (SharedRecording, TestHooks) {
    recording_hooks_with(GpuProfile::UNKNOWN)
}

// Hooks with an explicit GPU profile, for tests exercising the quality-preset
// ceiling.
pub(crate) fn recording_hooks_with(gpu_profile: GpuProfile) -> (SharedRecording, TestHooks) {
    let state = SharedRecording::default();
    let factory_state = Arc::clone(&state);
    let backend_factory: BackendFactory = Box::new(move |init: BackendInit<'_>| {
        Ok(Box::new(RecordingBackend::from_init(
            Arc::clone(&factory_state),
            init,
        )) as Box<dyn RenderBackend>)
    });
    (
        state,
        TestHooks {
            gpu_profile,
            backend_factory,
        },
    )
}

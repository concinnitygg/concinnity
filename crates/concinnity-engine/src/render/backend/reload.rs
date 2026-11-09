//! Replacing what a live backend runs without rebuilding it: the shaders it
//! was built from, or the whole world on the retained device and window.

use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::backend::init::{BackendInit, SwapchainConfig};

/// The in-place reloads a live backend can make.
///
/// Defaulted: a backend without them reports no reload flag and no hot-swap
/// configuration, which routes a world swap to a full rebuild.
pub trait WorldReload {
    /// Shared atomic flag the backend polls at frame start to trigger a
    /// shader rebuild. `Some` only under `cn run -f` on backends that ship
    /// hot-reload; `None` on production runs and on backends that do not. The
    /// debug server reads this to forward `reload-shaders` commands; the
    /// filesystem watcher writes it directly. Default: `None`.
    fn shader_reload_flag(&self) -> Option<alloc::sync::Arc<core::sync::atomic::AtomicBool>> {
        None
    }

    /// The swapchain-level configuration this live backend can hot-swap a world
    /// onto, or `None` when the backend cannot reload a world in place (it must
    /// be fully rebuilt instead). Read by the renderer when a transplanted
    /// backend is handed a new world (the `cn edit` live SAVE): the swap reuses
    /// the backend via [`Self::reload_world`] only when this equals the new
    /// world's `BackendInit::swapchain_config`; a `None` or a mismatch routes to a
    /// full rebuild (recreating the window). Default `None`: a backend without a
    /// real `reload_world` always rebuilds.
    fn hot_swap_config(&self) -> Option<SwapchainConfig> {
        None
    }

    /// Re-upload a new world's GPU content onto this already-constructed backend,
    /// reusing the live device + window + swapchain instead of building a new one.
    /// Driven by the `cn edit` live SAVE: after a structural edit recompiles the
    /// blobs, the renderer transplants the running backend into the rebuilt
    /// world and calls this so the edit applies without recreating the OS window
    /// or re-initializing the GPU device. The backend waits for the GPU to idle,
    /// drops the old world's content resources, and rebuilds them from `init` on
    /// the retained hardware. Only ever called when [`Self::hot_swap_config`]
    /// reported a config matching `init.swapchain_config()`, so the swapchain
    /// (pixel format / frames-in-flight / EDR) is guaranteed unchanged. Default
    /// [`RenderError::Unsupported`], which routes the swap to a full rebuild.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend that cannot reload in
    /// place, or an error if rebuilding the world's GPU content fails.
    fn reload_world(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        let _ = init;
        Err(RenderError::Unsupported { op: "reload_world" })
    }
}

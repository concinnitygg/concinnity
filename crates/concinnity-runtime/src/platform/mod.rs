//! Which graphics API this process renders with, and so which shader platform
//! it cooks for and consumes.
//!
//! The backend is this crate's own compile-time choice, so the crate that knows
//! it is the one that states it: the editor and the cook are handed the value
//! rather than resolving a backend of their own.
//!
//! A build compiles in the target's native backend and, with the `vulkan`
//! feature, Vulkan beside it. One of them is chosen per process, before
//! anything cooks or renders: [`select`] takes an explicit request,
//! [`select_persisted`] the player's settings-menu choice, and the first read of
//! [`current`] without either fixes the native backend. The runtime, the editor
//! and the cook all read [`current`], so a world built in a process plays in it.

pub(crate) mod device;

use std::sync::OnceLock;

use concinnity_core::platform::select::{native_platform, resolve};
use concinnity_core::platform::{GraphicsApiStatus, Platform, PlatformSet};
use concinnity_core::settings::api_trial::Step;
use concinnity_host::store::paths::StateTree;

static SELECTED: OnceLock<Platform> = OnceLock::new();

// The step this process's launch took on the persisted graphics API trial. The
// first launch decision of a process is the one that stands.
static LAUNCH_STEP: OnceLock<Step> = OnceLock::new();

/// The graphics APIs this build can render with, the target's native one
/// first. Empty on a build with no backend.
pub fn compiled() -> &'static [Platform] {
    #[cfg(any(backend_metal, backend_dx, backend_vk))]
    {
        concinnity_device::COMPILED_PLATFORMS
    }
    #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
    {
        &[]
    }
}

/// [`compiled`] as a set, or the native platform alone on a build with no
/// backend: what a cook writing blobs for this build targets.
pub fn compiled_set() -> PlatformSet {
    match compiled() {
        [] => native().into(),
        platforms => platforms.iter().copied().collect(),
    }
}

/// The graphics API this process renders with: the one [`select`] chose, else
/// the native one.
///
/// A build with no backend runs nothing that consumes bytecode, but the cook
/// still needs a language to produce; the one this target renders with is the
/// useful answer.
pub fn current() -> Platform {
    *SELECTED.get_or_init(native)
}

/// Choose the graphics API this process renders with, from `requested` (the
/// native one when `None`). A request the build did not compile in, or one
/// with no usable device on this machine, falls back to the native API with a
/// warning. Returns what the process renders with.
///
/// The choice is made once per process: call this before anything cooks or
/// renders. A later call, or one after [`current`] was read, keeps the earlier
/// answer and warns when it differs.
pub fn select(requested: Option<Platform>) -> Platform {
    let selection = resolve(requested, native(), compiled(), |platform| {
        device::probe_device_on(platform).is_some()
    });
    if let Some(fallback) = selection.fallback {
        tracing::warn!("{fallback}; rendering with {}", selection.platform.key());
    }
    let chosen = *SELECTED.get_or_init(|| selection.platform);
    crate::crash::note("backend", chosen.key());
    if chosen != selection.platform {
        tracing::warn!(
            "graphics API {} was already fixed for this process; keeping it over {}",
            chosen.key(),
            selection.platform.key()
        );
    }
    chosen
}

/// Where this launch stands on the graphics API chosen in the settings menu.
pub(crate) fn api_status() -> GraphicsApiStatus {
    GraphicsApiStatus::from_launch(current(), LAUNCH_STEP.get().copied())
}

/// Choose the graphics API from the settings-menu choice persisted in `tree`,
/// as [`select`] would from a request. Does nothing when this process already
/// chose (an explicit request outranks the persisted choice).
///
/// A newly chosen API is tried and kept only once this launch draws a frame on
/// it; when the last launch on it never did, this launch starts on the native
/// API and drops the choice.
pub fn select_persisted(tree: Option<&StateTree>) -> Platform {
    if let Some(&chosen) = SELECTED.get() {
        return chosen;
    }
    select(persisted_request(tree))
}

// The API the settings in `tree` ask this launch for, after moving the choice's
// trial along (and saving it) the way the launch plan says.
fn persisted_request(tree: Option<&StateTree>) -> Option<Platform> {
    let mut settings = crate::persist::settings::load(tree);
    let launch = concinnity_core::settings::api_trial::plan(
        settings.graphics.graphics_api,
        settings.graphics.graphics_api_trial,
    );
    if let Some(step) = launch.step {
        concinnity_core::settings::api_trial::apply(step, &mut settings.graphics);
        let _ = LAUNCH_STEP.set(step);
        if let Err(e) = crate::persist::settings::save(&settings, tree) {
            tracing::warn!("graphics API choice: could not save settings: {e}");
        }
    }
    launch.request
}

// What the target renders with when nothing is selected.
fn native() -> Platform {
    compiled()
        .first()
        .copied()
        .unwrap_or_else(|| native_platform(std::env::consts::OS))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A launch that never confirms its new choice leaves it trying, and the
    // launch after it gives the choice up.
    #[test]
    fn an_unconfirmed_choice_is_tried_once_then_dropped() {
        use concinnity_core::settings::GraphicsApiTrial;

        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        let mut settings = concinnity_core::settings::StoredSettings::default();
        settings.graphics.graphics_api = Some(Platform::Vulkan);
        settings.graphics.graphics_api_trial = Some(GraphicsApiTrial::Untried);
        crate::persist::settings::save(&settings, Some(&tree)).unwrap();

        assert_eq!(persisted_request(Some(&tree)), Some(Platform::Vulkan));
        let stored = crate::persist::settings::load(Some(&tree)).graphics;
        assert_eq!(stored.graphics_api_trial, Some(GraphicsApiTrial::Trying));

        assert_eq!(persisted_request(Some(&tree)), None);
        let stored = crate::persist::settings::load(Some(&tree)).graphics;
        assert_eq!(stored.graphics_api, None);
        assert_eq!(stored.graphics_api_trial, None);
    }

    #[test]
    fn a_confirmed_choice_is_requested_without_a_write() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        let mut settings = concinnity_core::settings::StoredSettings::default();
        settings.graphics.graphics_api = Some(Platform::Vulkan);
        crate::persist::settings::save(&settings, Some(&tree)).unwrap();
        let before = std::fs::read(tree.settings_path()).unwrap();

        assert_eq!(persisted_request(Some(&tree)), Some(Platform::Vulkan));
        assert_eq!(std::fs::read(tree.settings_path()).unwrap(), before);
    }

    #[test]
    fn the_native_api_leads_what_the_build_compiled() {
        assert_eq!(compiled().first().copied().unwrap_or_else(native), native());
    }
}

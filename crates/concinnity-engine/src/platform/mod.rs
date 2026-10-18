//! Which graphics API this process renders with, and so which shader platform
//! it cooks for and consumes.
//!
//! A build compiles in the target's native backend and, with the `vulkan`
//! feature, Vulkan beside it. One of them is chosen per process, before
//! anything cooks or renders: [`select`] takes an explicit request,
//! [`select_persisted`] the player's settings-menu choice, and the first read of
//! [`current`] without either fixes the native backend. The runtime, the editor
//! and the cook all read [`current`], so a world built in a process plays in it.

use std::sync::OnceLock;

use concinnity_core::platform::{Platform, PlatformSet};
use concinnity_host::store::paths::StateTree;

mod trial;

pub(crate) use trial::{confirm_pending, reverted};

static SELECTED: OnceLock<Platform> = OnceLock::new();

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
        crate::device::probe_device_on(platform).is_some()
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
    let mut settings = crate::settings::store::load(tree);
    let launch = trial::plan(
        settings.graphics.graphics_api,
        settings.graphics.graphics_api_trial,
    );
    if let Some(step) = launch.step {
        trial::apply(step, &mut settings.graphics);
        trial::record(step);
        if let Err(e) = crate::settings::store::save(&settings, tree) {
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

// What a target would render with when no backend is compiled in.
fn native_platform(target_os: &str) -> Platform {
    match target_os {
        "macos" | "ios" => Platform::Metal,
        "windows" => Platform::DirectX,
        _ => Platform::Vulkan,
    }
}

// The API a request resolves to, and why it was not the one asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Selection {
    platform: Platform,
    fallback: Option<Fallback>,
}

// Why a requested API gave way to the native one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fallback {
    NotCompiled(Platform),
    NoDevice(Platform),
}

impl std::fmt::Display for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fallback::NotCompiled(p) => write!(f, "graphics API {} is not in this build", p.key()),
            Fallback::NoDevice(p) => write!(f, "graphics API {} found no usable device", p.key()),
        }
    }
}

// Resolve `requested` against what the build compiled in and what `usable` says
// the machine runs. The native API is never probed here: it is the fallback,
// and a machine with no GPU for it resolves to headless later on its own.
fn resolve(
    requested: Option<Platform>,
    native: Platform,
    compiled: &[Platform],
    usable: impl FnOnce(Platform) -> bool,
) -> Selection {
    let fallback = match requested {
        None => None,
        Some(platform) if platform == native => None,
        Some(platform) if !compiled.contains(&platform) => Some(Fallback::NotCompiled(platform)),
        Some(platform) if !usable(platform) => Some(Fallback::NoDevice(platform)),
        Some(platform) => {
            return Selection {
                platform,
                fallback: None,
            };
        }
    };
    Selection {
        platform: native,
        fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DX_VK: &[Platform] = &[Platform::DirectX, Platform::Vulkan];

    fn unreachable_probe(_: Platform) -> bool {
        panic!("the native API is never probed")
    }

    #[test]
    fn every_target_has_a_native_platform() {
        assert_eq!(native_platform("macos"), Platform::Metal);
        assert_eq!(native_platform("windows"), Platform::DirectX);
        assert_eq!(native_platform("linux"), Platform::Vulkan);
        assert_eq!(native_platform("ios"), Platform::Metal);
        assert_eq!(native_platform("android"), Platform::Vulkan);
    }

    #[test]
    fn no_request_or_the_native_one_renders_native_without_probing() {
        for requested in [None, Some(Platform::DirectX)] {
            let got = resolve(requested, Platform::DirectX, DX_VK, unreachable_probe);
            assert_eq!(got.platform, Platform::DirectX);
            assert_eq!(got.fallback, None);
        }
    }

    #[test]
    fn a_compiled_api_with_a_device_is_taken() {
        let got = resolve(Some(Platform::Vulkan), Platform::DirectX, DX_VK, |_| true);
        assert_eq!(got.platform, Platform::Vulkan);
        assert_eq!(got.fallback, None);
    }

    #[test]
    fn an_api_the_build_left_out_falls_back_native() {
        let got = resolve(
            Some(Platform::Metal),
            Platform::DirectX,
            DX_VK,
            unreachable_probe,
        );
        assert_eq!(got.platform, Platform::DirectX);
        assert_eq!(got.fallback, Some(Fallback::NotCompiled(Platform::Metal)));
    }

    #[test]
    fn an_api_with_no_device_falls_back_native() {
        let got = resolve(Some(Platform::Vulkan), Platform::DirectX, DX_VK, |_| false);
        assert_eq!(got.platform, Platform::DirectX);
        assert_eq!(got.fallback, Some(Fallback::NoDevice(Platform::Vulkan)));
    }

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
        crate::settings::store::save(&settings, Some(&tree)).unwrap();

        assert_eq!(persisted_request(Some(&tree)), Some(Platform::Vulkan));
        let stored = crate::settings::store::load(Some(&tree)).graphics;
        assert_eq!(stored.graphics_api_trial, Some(GraphicsApiTrial::Trying));

        assert_eq!(persisted_request(Some(&tree)), None);
        let stored = crate::settings::store::load(Some(&tree)).graphics;
        assert_eq!(stored.graphics_api, None);
        assert_eq!(stored.graphics_api_trial, None);
    }

    #[test]
    fn a_confirmed_choice_is_requested_without_a_write() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        let mut settings = concinnity_core::settings::StoredSettings::default();
        settings.graphics.graphics_api = Some(Platform::Vulkan);
        crate::settings::store::save(&settings, Some(&tree)).unwrap();
        let before = std::fs::read(tree.settings_path()).unwrap();

        assert_eq!(persisted_request(Some(&tree)), Some(Platform::Vulkan));
        assert_eq!(std::fs::read(tree.settings_path()).unwrap(), before);
    }

    #[test]
    fn the_native_api_leads_what_the_build_compiled() {
        assert_eq!(compiled().first().copied().unwrap_or(native()), native());
    }
}

//! Process-wide flags shared between the engine loop (library) and the
//! binary-only `cn debug` subsystem. Only the flags the library itself names
//! live here; the world.jsonl / shader-stage "changed" flags and the decal /
//! emitter spawn queue moved fully into the binary-only debug tree
//! (`crate::debug`), since nothing in the library references them.
//!
//!   ENABLED              "are we running under a dev-loop entry point?" Set
//!                        once by main.rs's `Commands::Debug` / `Commands::Editor`
//!                        arms before world build; read by `GraphicsSystem::init`
//!                        / `AnimationSystem` / the draw list builder to enable
//!                        disk-first shader loading + the hot-reload source
//!                        capture. `cn run` leaves it false so production keeps
//!                        the static `include_str!`-baked path with no
//!                        filesystem dependency.
//!   PENDING_ANIMATIONS   "an Animation source changed." Set by the cn debug
//!                        watcher / WS `reload-assets` handler; consumed by the
//!                        editor crate's `anim_reload::reload_clips_if_pending`,
//!                        which the debug drive calls each frame to re-import
//!                        file-backed clips. The flag lives here (in the runtime
//!                        crate) because it bridges the runtime AnimationSystem,
//!                        which reads ENABLED, and the editor-driven hot-reload.
//!   VALIDATION           "did the launch request graphics validation?" Set by
//!                        the CLI `--validation` flag (`cn run` / `cn debug`).
//!                        Tri-state: unset defers to the build profile (on for
//!                        debug, off for release). `resolve_validation` settles
//!                        the two, and `GraphicsSystem::init` enables the
//!                        DirectX / Vulkan debug layers from the result. Metal's
//!                        validation layer cannot be toggled from a running
//!                        process, so the CLI re-execs with the env var instead;
//!                        this flag does not drive Metal.
//!   QUALITY_PRESET       "did the launch force a master quality preset?" Set by
//!                        the CLI `--quality-preset` flag. Outranks the persisted
//!                        settings-menu choice at `GraphicsSystem::init` and is
//!                        never written back, so a probe / CI run can force a
//!                        preset (e.g. `ultra`, the only tier whose ceiling
//!                        permits ray-traced reflections) without touching
//!                        settings.bin. Unset leaves the persisted choice.
//!   RT_DYNAMIC           "how should the ray-tracing acceleration structure
//!                        track moving props?" Set by the CLI `--rt-dynamic`
//!                        flag; travels to the backends through
//!                        `PostSettings::rt_dynamic`. Unset resolves to `Auto`,
//!                        the shipping dirty-gated rebuild.
//!   RT_SKINNED_GEOMETRY  "may skinned meshes join the ray-tracing acceleration
//!                        structure?" Set by the CLI `--rt-skinned-geometry`
//!                        flag; travels to the backends through
//!                        `PostSettings::rt_skinned_geometry`. Unset leaves them
//!                        in, so clearing it isolates the skinned trace path.
//!   WORLD_JSONL_PATH     the world.jsonl the dev host is running. Set by the
//!                        editor's `cn debug` / `cn editor` entry once the world
//!                        path is resolved; read by `GraphicsSystem::init` (only
//!                        under ENABLED) so the Prop-transform hot-reload watcher
//!                        knows which file to subscribe to. world.jsonl discovery
//!                        is authoring I/O that lives in `concinnity-cook`, which
//!                        the runtime does not link, so the dev host resolves the
//!                        path and hands it in rather than the engine looking it
//!                        up. Left None for `cn run` and embedded preview.
//!
//! A static is the pragmatic shape here: the flags are process-wide because the
//! rendering backend is too (a single context per process owns the GPU), and
//! plumbing them through the public `App` / `run_interpreted` signatures would
//! touch far more code for the same observable behaviour.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub use crate::gfx::quality_preset::QualityPreset;
pub use concinnity_core::render::rt_geom::RtDynamicMode;

use crate::gfx::quality_preset::{preset_at, preset_index};

static ENABLED: AtomicBool = AtomicBool::new(false);
static PENDING_ANIMATIONS: AtomicBool = AtomicBool::new(false);
// "keep the presented frame blit-readable for an exit screenshot." Set by
// `cn run --screenshot` before world build; read by `GraphicsSystem::init`.
// The dev loop's ENABLED implies capture without this flag.
static CAPTURE: AtomicBool = AtomicBool::new(false);

// Tri-state validation request: 0 = unset (use the build-profile default),
// 1 = explicitly off, 2 = explicitly on.
static VALIDATION: AtomicU8 = AtomicU8::new(0);

// Launch-forced master quality preset: 0 = unset, otherwise the preset's cycle
// index plus one.
static QUALITY_PRESET: AtomicU8 = AtomicU8::new(0);

// Launch-forced ray-tracing update mode: 0 = unset, otherwise `RT_DYNAMIC_ORDER`
// index plus one.
static RT_DYNAMIC: AtomicU8 = AtomicU8::new(0);

// The modes `RT_DYNAMIC` encodes, in encoding order.
const RT_DYNAMIC_ORDER: [RtDynamicMode; 4] = [
    RtDynamicMode::Off,
    RtDynamicMode::Auto,
    RtDynamicMode::Rebuild,
    RtDynamicMode::Tlas,
];

// Tri-state skinned-RT-geometry request: 0 = unset, 1 = excluded, 2 = included.
static RT_SKINNED_GEOMETRY: AtomicU8 = AtomicU8::new(0);

// Path to the world.jsonl the dev host is running, or None outside a dev host.
static WORLD_JSONL_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Mark this process as running under a dev-loop entry point. Call once
/// before world build; the library only reads the flag.
pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::SeqCst);
}

// True when the process is running under a dev-loop entry point that wants
// shader hot-reload. False for `cn run` and any embedded preview.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

// Arm frame capture for a production run that wants an exit screenshot.
// Called by `start_runtime` before world build when a screenshot path was
// requested.
pub(crate) fn set_capture(v: bool) {
    CAPTURE.store(v, Ordering::SeqCst);
}

// True when a production run armed frame capture (`cn run --screenshot`).
pub(crate) fn capture() -> bool {
    CAPTURE.load(Ordering::SeqCst)
}

/// Raise the "Animation source changed" flag. Called by the asset hot-reload
/// watcher and the WS `reload-assets` handler; the library only reads it.
pub fn set_pending_animations() {
    PENDING_ANIMATIONS.store(true, Ordering::SeqCst);
}

/// Swap the "Animation source changed" flag to `false`, returning whether it
/// was set. The editor crate's `anim_reload::reload_clips_if_pending` calls
/// this; a `true` result kicks the per-clip re-import pass.
pub fn take_pending_animations() -> bool {
    PENDING_ANIMATIONS.swap(false, Ordering::SeqCst)
}

/// Record the CLI `--validation` request. `None` leaves the build-profile
/// default in effect; `Some` forces validation on or off. The library only
/// reads it.
pub fn set_validation(v: Option<bool>) {
    let encoded = match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    VALIDATION.store(encoded, Ordering::SeqCst);
}

// The CLI validation request, or `None` when the launch did not specify one.
pub(crate) fn validation() -> Option<bool> {
    match VALIDATION.load(Ordering::SeqCst) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

// Settle the graphics-validation request: the CLI `--validation` flag if the
// launch passed one, otherwise the build profile. Running a debug layer is a
// launch concern, so no world can ask for it.
pub(crate) fn resolve_validation() -> bool {
    validation().unwrap_or(cfg!(debug_assertions))
}

/// Record the CLI `--quality-preset` request. `None` leaves the persisted
/// settings-menu choice in effect. The library only reads it.
pub fn set_quality_preset(preset: Option<QualityPreset>) {
    let encoded = preset.map_or(0, |p| preset_index(p) as u8 + 1);
    QUALITY_PRESET.store(encoded, Ordering::SeqCst);
}

// The CLI quality-preset request, or `None` when the launch did not force one.
pub(crate) fn quality_preset() -> Option<QualityPreset> {
    match QUALITY_PRESET.load(Ordering::SeqCst) {
        0 => None,
        n => Some(preset_at(n as usize - 1)),
    }
}

// Settle the master quality preset: the CLI `--quality-preset` flag if the
// launch passed one, otherwise the persisted settings-menu choice. `None` means
// neither exists, which is a first launch the caller seeds.
pub(crate) fn resolve_quality_preset(persisted: Option<QualityPreset>) -> Option<QualityPreset> {
    quality_preset().or(persisted)
}

/// Record the CLI `--rt-dynamic` request. `None` leaves the default `Auto`
/// update mode in effect. The library only reads it.
pub fn set_rt_dynamic(mode: Option<RtDynamicMode>) {
    let encoded = mode.map_or(0, |m| {
        RT_DYNAMIC_ORDER
            .iter()
            .position(|&candidate| candidate == m)
            .expect("RT_DYNAMIC_ORDER covers every mode") as u8
            + 1
    });
    RT_DYNAMIC.store(encoded, Ordering::SeqCst);
}

// The CLI ray-tracing update-mode request, or `None` when the launch passed one.
pub(crate) fn rt_dynamic() -> Option<RtDynamicMode> {
    RT_DYNAMIC_ORDER
        .get(RT_DYNAMIC.load(Ordering::SeqCst).wrapping_sub(1) as usize)
        .copied()
}

// Settle how the acceleration structure tracks moving props: the CLI
// `--rt-dynamic` flag if the launch passed one, otherwise `Auto`.
pub(crate) fn resolve_rt_dynamic() -> RtDynamicMode {
    rt_dynamic().unwrap_or_default()
}

/// Record the CLI `--rt-skinned-geometry` request. `None` leaves skinned meshes
/// in the acceleration structure. The library only reads it.
pub fn set_rt_skinned_geometry(v: Option<bool>) {
    let encoded = match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    RT_SKINNED_GEOMETRY.store(encoded, Ordering::SeqCst);
}

// The CLI skinned-RT-geometry request, or `None` when the launch did not pass one.
pub(crate) fn rt_skinned_geometry() -> Option<bool> {
    match RT_SKINNED_GEOMETRY.load(Ordering::SeqCst) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

// Settle whether skinned meshes join the acceleration structure: the CLI
// `--rt-skinned-geometry` flag if the launch passed one, otherwise in.
pub(crate) fn resolve_rt_skinned_geometry() -> bool {
    rt_skinned_geometry().unwrap_or(true)
}

/// Record the world.jsonl path the dev host resolved, so the hot-reload watcher
/// can subscribe to it. Called by the editor's `cn debug` / `cn editor` entry
/// before world build; the library only reads it.
pub fn set_world_jsonl_path(path: Option<String>) {
    *WORLD_JSONL_PATH.lock().unwrap() = path;
}

// The world.jsonl path the dev host handed in, or None outside a dev host. Read
// by `GraphicsSystem::init` to seed the Prop-transform reload watcher.
pub(crate) fn world_jsonl_path() -> Option<String> {
    WORLD_JSONL_PATH.lock().unwrap().clone()
}

// Shared flag access for a test whose code path reads a flag. Held for as long
// as the read matters: for graphics init, across the whole `run_init`.
//
// The lock is the workspace's one process-global lock rather than a private
// static, so a flag written here cannot race a test in another crate that
// reaches these same flags through a different guard.
#[cfg(test)]
pub(crate) fn read_access() -> concinnity_testing::SharedAccess {
    concinnity_testing::shared()
}

// Exclusive flag access for a test that writes one. Restores every flag graphics
// init reads when it drops, so a panicking test cannot leak one into the rest of
// the binary. Poison is ignored: the test holding it has already failed, and
// erroring every later lock buries that failure under a cascade. Not reentrant,
// so a test holding this must not also take `read_access`.
#[cfg(test)]
pub(crate) struct WriteAccess {
    _guard: concinnity_testing::ExclusiveAccess,
    enabled: bool,
    validation: Option<bool>,
    quality_preset: Option<QualityPreset>,
    rt_dynamic: Option<RtDynamicMode>,
    rt_skinned_geometry: Option<bool>,
    world_jsonl_path: Option<String>,
}

#[cfg(test)]
pub(crate) fn write_access() -> WriteAccess {
    WriteAccess {
        _guard: concinnity_testing::exclusive(),
        enabled: enabled(),
        validation: validation(),
        quality_preset: quality_preset(),
        rt_dynamic: rt_dynamic(),
        rt_skinned_geometry: rt_skinned_geometry(),
        world_jsonl_path: world_jsonl_path(),
    }
}

#[cfg(test)]
impl Drop for WriteAccess {
    fn drop(&mut self) {
        set_enabled(self.enabled);
        set_validation(self.validation);
        set_quality_preset(self.quality_preset);
        set_rt_dynamic(self.rt_dynamic);
        set_rt_skinned_geometry(self.rt_skinned_geometry);
        set_world_jsonl_path(self.world_jsonl_path.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_off_and_round_trips() {
        let _flags = write_access();
        set_enabled(false);
        assert!(!enabled());
        set_enabled(true);
        assert!(enabled());
    }

    #[test]
    fn validation_tristate_round_trips() {
        let _flags = write_access();
        set_validation(None);
        assert_eq!(validation(), None);
        set_validation(Some(true));
        assert_eq!(validation(), Some(true));
        set_validation(Some(false));
        assert_eq!(validation(), Some(false));
    }

    #[test]
    fn every_quality_preset_round_trips_through_the_flag() {
        let _flags = write_access();
        set_quality_preset(None);
        assert_eq!(quality_preset(), None);
        for preset in QualityPreset::ALL {
            set_quality_preset(Some(preset));
            assert_eq!(quality_preset(), Some(preset));
        }
    }

    #[test]
    fn the_quality_flag_outranks_the_persisted_choice() {
        let _flags = write_access();

        // No flag: the persisted settings-menu choice decides, unchanged.
        set_quality_preset(None);
        assert_eq!(resolve_quality_preset(None), None);
        assert_eq!(
            resolve_quality_preset(Some(QualityPreset::Auto)),
            Some(QualityPreset::Auto)
        );

        // The flag wins over any persisted value, and over none.
        set_quality_preset(Some(QualityPreset::Ultra));
        assert_eq!(
            resolve_quality_preset(Some(QualityPreset::Auto)),
            Some(QualityPreset::Ultra)
        );
        assert_eq!(resolve_quality_preset(None), Some(QualityPreset::Ultra));
    }

    #[test]
    fn every_rt_dynamic_mode_round_trips_and_unset_is_auto() {
        let _flags = write_access();
        set_rt_dynamic(None);
        assert_eq!(rt_dynamic(), None);
        assert_eq!(resolve_rt_dynamic(), RtDynamicMode::Auto);
        for mode in RT_DYNAMIC_ORDER {
            set_rt_dynamic(Some(mode));
            assert_eq!(rt_dynamic(), Some(mode));
            assert_eq!(resolve_rt_dynamic(), mode);
        }
    }

    #[test]
    fn skinned_rt_geometry_is_in_unless_the_flag_clears_it() {
        let _flags = write_access();
        set_rt_skinned_geometry(None);
        assert_eq!(rt_skinned_geometry(), None);
        assert!(resolve_rt_skinned_geometry());
        set_rt_skinned_geometry(Some(true));
        assert!(resolve_rt_skinned_geometry());
        set_rt_skinned_geometry(Some(false));
        assert!(!resolve_rt_skinned_geometry());
    }

    #[test]
    fn the_launch_flag_outranks_the_build_profile() {
        let _flags = write_access();

        // No flag: the build profile decides.
        set_validation(None);
        assert_eq!(resolve_validation(), cfg!(debug_assertions));

        // An explicit flag decides instead, either way.
        set_validation(Some(false));
        assert!(!resolve_validation());
        set_validation(Some(true));
        assert!(resolve_validation());
    }
}

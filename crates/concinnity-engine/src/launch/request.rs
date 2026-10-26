use alloc::string::{String, ToString};
use core::num::NonZeroU32;

use super::LaunchArgs;
use crate::ecs::{Resources, ViewOverrides, World};
use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::render::output::dlss::DlssPreset;
use crate::render::rt_accel::geom::RtDynamicMode;
use crate::settings::{QualityPreset, StoredSettings, UserGraphicsSettings};

/// Where screenshots land unless the launch names another directory: relative to the working directory the run started in.
pub const DEFAULT_SCREENSHOTS_DIR: &str = "screenshots";

/// The file a capture at `path` is written to for a run whose screenshots go
/// under `dir`: `path` itself when it is absolute, else `path` under `dir`.
pub fn screenshot_path(dir: &str, path: &str) -> String {
    if is_absolute(path) {
        return path.to_string();
    }
    let dir = dir.trim_end_matches(['/', '\\']);
    if dir.is_empty() {
        return path.to_string();
    }
    alloc::format!("{dir}/{path}")
}

// Whether `path` is absolute on any host a run writes on: rooted (`/`, `\\`)
// or led by a drive letter (`C:`).
fn is_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    matches!(bytes.first(), Some(b'/' | b'\\'))
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
}

/// What the launch asked the host to arm, published as a world resource before
/// the world starts. A run without one runs the shipping behavior. Each `None`
/// defers to that setting's default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaunchRequest {
    /// Keep the presented frame blit-readable for an exit screenshot.
    pub capture: bool,
    /// Run as a development session: capture hot-reload sources, load shaders
    /// disk-first, keep frame capture available, and show the debug HUD.
    pub dev_loop: bool,
    /// What the launch's flags ask of the run: a benchmark, vsync, a frame-rate
    /// cap, a quality preset or a frame count. An app's own
    /// [`LaunchArgs`] resource, published before start, replaces these.
    pub args: LaunchArgs,
    /// Force the DirectX / Vulkan debug layers on or off. `None` follows the
    /// build profile. Metal's layer is set by the process environment instead.
    pub validation: Option<bool>,
    /// Force how the ray-tracing acceleration structure tracks moving props.
    /// `None` is [`RtDynamicMode::Auto`].
    pub rt_dynamic: Option<RtDynamicMode>,
    /// Whether skinned meshes join the ray-tracing acceleration structure.
    /// `None` leaves them in.
    pub rt_skinned_geometry: Option<bool>,
    /// Force the DLSS render preset over the persisted choice, unpersisted.
    pub dlss_preset: Option<DlssPreset>,
    /// Step the frame clock by exactly `1 / rate` seconds per frame instead of
    /// the wall time between frames, so a given frame sees the same clock on
    /// every launch. `None` follows the wall clock.
    pub fixed_frame_rate: Option<NonZeroU32>,
    /// The feature passes the run starts with, as the editor's View menu shows
    /// them. Every flag set is the shipping frame.
    pub show: ShowFlags,
}

impl LaunchRequest {
    /// Publish the request into `world` before it starts, with what it arms:
    /// the world's own [`LaunchArgs`] replace the request's, a benchmark run is
    /// armed, and the view overrides are inserted.
    pub fn publish(&mut self, world: &mut World) {
        if let Some(args) = world.remove_resource::<LaunchArgs>() {
            self.args = args;
        }
        world.insert_resource(self.clone());
        if self.args.benchmark {
            crate::benchmark::arm(world);
        }
        if let Some(view) = self.view_overrides() {
            world.insert_resource(view);
        }
    }

    /// The view state the run starts with, or `None` for the shipping frame.
    fn view_overrides(&self) -> Option<ViewOverrides> {
        (self.show != ShowFlags::all()).then_some(ViewOverrides {
            mode: ViewMode::Lit,
            show: self.show,
        })
    }

    /// The directory this run writes screenshots under.
    pub fn screenshots_dir(&self) -> &str {
        self.args
            .screenshots_dir
            .as_deref()
            .unwrap_or(DEFAULT_SCREENSHOTS_DIR)
    }

    /// Whether the graphics debug layers run: the request, else the build profile.
    pub fn resolve_validation(&self) -> bool {
        self.validation.unwrap_or(cfg!(debug_assertions))
    }

    /// The master quality preset: the request (never persisted), else the
    /// `stored` choice. A first launch, with neither, seeds `Auto` into the
    /// store published in `resources` so it persists once.
    pub fn resolve_quality_preset(
        &self,
        stored: &StoredSettings,
        resources: &mut Resources,
    ) -> QualityPreset {
        if let Some(preset) = self.args.quality_preset.or(stored.graphics.quality_preset) {
            return preset;
        }
        let mut seeded = stored.clone();
        seeded.graphics.quality_preset = Some(QualityPreset::Auto);
        seeded.store_changed(resources);
        QualityPreset::Auto
    }

    /// How the acceleration structure tracks moving props: the request, else `Auto`.
    pub fn resolve_rt_dynamic(&self) -> RtDynamicMode {
        self.rt_dynamic.unwrap_or_default()
    }

    /// The settings this launch overrides for its own run, layered over the
    /// user's persisted choices and never persisted: vsync, the frame-rate cap,
    /// the window mode and the DLSS preset.
    pub fn session_overrides(&self) -> UserGraphicsSettings {
        UserGraphicsSettings {
            vsync: self.args.vsync.then_some(true),
            fps_cap: self.args.fps_cap,
            window_mode: self.args.window_mode,
            dlss_preset: self.dlss_preset,
            ..UserGraphicsSettings::default()
        }
    }

    /// Whether skinned meshes join the acceleration structure: the request, else in.
    pub fn resolve_rt_skinned_geometry(&self) -> bool {
        self.rt_skinned_geometry.unwrap_or(true)
    }

    /// Whether the presented frame stays readable: always in a dev session, and
    /// for a launch that asked for an exit screenshot.
    pub fn frame_capture(&self) -> bool {
        self.dev_loop || self.capture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The world's own arguments win, and the published request carries them.
    #[test]
    fn publishing_adopts_the_worlds_launch_args() {
        let mut world = World::new();
        world.insert_resource(LaunchArgs {
            frames: Some(3),
            ..Default::default()
        });
        let mut request = LaunchRequest::default();
        request.publish(&mut world);
        assert_eq!(request.args.frames, Some(3));
        assert_eq!(world.resource::<LaunchRequest>(), Some(&request));
        assert!(world.resource::<LaunchArgs>().is_none());
        assert!(world.resource::<ViewOverrides>().is_none());
    }

    #[test]
    fn screenshots_go_under_the_launch_s_directory_or_the_default() {
        let mut request = LaunchRequest::default();
        assert_eq!(request.screenshots_dir(), DEFAULT_SCREENSHOTS_DIR);
        request.args.screenshots_dir = Some("shots/dx12".into());
        assert_eq!(request.screenshots_dir(), "shots/dx12");
    }

    // A relative capture lands under the directory; an absolute one, on any
    // host's form, stays where it says.
    #[test]
    fn a_capture_path_resolves_under_the_screenshot_directory() {
        assert_eq!(
            screenshot_path("screenshots", "benchmark/orbit.png"),
            "screenshots/benchmark/orbit.png"
        );
        assert_eq!(screenshot_path("shots/vk/", "a.png"), "shots/vk/a.png");
        assert_eq!(screenshot_path("", "a.png"), "a.png");
        for absolute in [
            "/tmp/a.png",
            "C:\\out\\a.png",
            "d:/a.png",
            "\\\\host\\a.png",
        ] {
            assert_eq!(screenshot_path("screenshots", absolute), absolute);
        }
    }

    #[test]
    fn the_validation_request_outranks_the_build_profile() {
        let request = |validation| LaunchRequest {
            validation,
            ..Default::default()
        };
        assert_eq!(request(None).resolve_validation(), cfg!(debug_assertions));
        assert!(!request(Some(false)).resolve_validation());
        assert!(request(Some(true)).resolve_validation());
    }

    // The request outranks the stored choice, which outranks a first launch's
    // seed; only the seed touches the store.
    #[test]
    fn the_quality_preset_request_outranks_the_stored_choice_and_the_seed() {
        use crate::settings::SettingsRevision;

        let mut resources = Resources::default();
        let first = StoredSettings::default();
        let unset = LaunchRequest::default();
        assert_eq!(
            unset.resolve_quality_preset(&first, &mut resources),
            QualityPreset::Auto
        );
        let seeded = resources.get::<StoredSettings>().expect("seeded store");
        assert_eq!(seeded.graphics.quality_preset, Some(QualityPreset::Auto));
        assert_eq!(resources.get::<SettingsRevision>().map(|r| r.0), Some(1));

        let mut stored = StoredSettings::default();
        stored.graphics.quality_preset = Some(QualityPreset::Low);
        let mut untouched = Resources::default();
        assert_eq!(
            unset.resolve_quality_preset(&stored, &mut untouched),
            QualityPreset::Low
        );

        let forced = LaunchRequest {
            args: LaunchArgs {
                quality_preset: Some(QualityPreset::Ultra),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        assert_eq!(
            forced.resolve_quality_preset(&stored, &mut untouched),
            QualityPreset::Ultra
        );
        assert_eq!(
            forced.resolve_quality_preset(&first, &mut untouched),
            QualityPreset::Ultra
        );
        assert!(untouched.get::<StoredSettings>().is_none());
    }

    #[test]
    fn an_unset_rt_dynamic_request_is_auto() {
        assert_eq!(
            LaunchRequest::default().resolve_rt_dynamic(),
            RtDynamicMode::Auto
        );
        for mode in [
            RtDynamicMode::Off,
            RtDynamicMode::Auto,
            RtDynamicMode::Rebuild,
            RtDynamicMode::Tlas,
        ] {
            let request = LaunchRequest {
                rt_dynamic: Some(mode),
                ..Default::default()
            };
            assert_eq!(request.resolve_rt_dynamic(), mode);
        }
    }

    #[test]
    fn vsync_and_a_frame_rate_cap_are_session_overrides() {
        let forced = LaunchRequest {
            args: LaunchArgs {
                vsync: true,
                fps_cap: Some(30),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        let user = UserGraphicsSettings {
            vsync: Some(false),
            fps_cap: Some(144),
            ..Default::default()
        };
        let merged = user.overlaid_by(&forced.session_overrides());
        assert_eq!(merged.vsync, Some(true));
        assert_eq!(merged.fps_cap, Some(30));
    }

    #[test]
    fn a_requested_dlss_preset_is_a_session_override() {
        assert_eq!(
            LaunchRequest::default().session_overrides(),
            UserGraphicsSettings::default()
        );
        let forced = LaunchRequest {
            dlss_preset: Some(DlssPreset::M),
            ..Default::default()
        };
        let user = UserGraphicsSettings {
            dlss_preset: Some(DlssPreset::K),
            vsync: Some(true),
            ..Default::default()
        };
        let merged = user.overlaid_by(&forced.session_overrides());
        assert_eq!(merged.dlss_preset, Some(DlssPreset::M));
        assert_eq!(merged.vsync, Some(true));
    }

    #[test]
    fn skinned_rt_geometry_is_in_unless_the_request_clears_it() {
        let request = |rt_skinned_geometry| LaunchRequest {
            rt_skinned_geometry,
            ..Default::default()
        };
        assert!(request(None).resolve_rt_skinned_geometry());
        assert!(request(Some(true)).resolve_rt_skinned_geometry());
        assert!(!request(Some(false)).resolve_rt_skinned_geometry());
    }

    // The shipping frame publishes no view state; a hidden pass starts the run
    // lit with that pass off.
    #[test]
    fn a_hidden_pass_starts_the_run_without_it() {
        assert_eq!(LaunchRequest::default().view_overrides(), None);
        let hidden = LaunchRequest {
            show: ShowFlags::all().toggled(ShowFlags::REACTIVE),
            ..Default::default()
        };
        let view = hidden.view_overrides().expect("view state");
        assert_eq!(view.mode, ViewMode::Lit);
        assert!(!view.show.contains(ShowFlags::REACTIVE));
        assert!(view.show.contains(ShowFlags::FOG));
    }

    #[test]
    fn frame_capture_is_armed_by_a_dev_loop_or_a_capture_request() {
        assert!(!LaunchRequest::default().frame_capture());
        let dev_loop = LaunchRequest {
            dev_loop: true,
            ..Default::default()
        };
        assert!(dev_loop.frame_capture());
        let capture = LaunchRequest {
            capture: true,
            ..Default::default()
        };
        assert!(capture.frame_capture());
    }
}

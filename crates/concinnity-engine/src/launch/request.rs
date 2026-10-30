use alloc::string::{String, ToString};

use super::LaunchArgs;
use crate::ecs::{Resources, ViewOverrides, World};
use crate::gfx::view_modes::{ShowFlags, ViewMode};
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
    /// Run as a development session: capture hot-reload sources, load shaders
    /// disk-first, keep frame capture available, and show the debug HUD.
    pub dev_loop: bool,
    /// What the launch's command and flags ask of the run. An app's own
    /// [`LaunchArgs`] resource, published before start, replaces these.
    pub args: LaunchArgs,
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
        if self.args.benchmarks() {
            crate::benchmark::arm(world, self.args.stop_at());
        }
        if let Some(view) = self.view_overrides() {
            world.insert_resource(view);
        }
        if self.args.hud == Some(false) {
            world.insert_resource(crate::ecs::HudHidden(true));
        }
    }

    /// The view state the run starts with, or `None` for the shipping frame.
    fn view_overrides(&self) -> Option<ViewOverrides> {
        (self.args.show != ShowFlags::all()).then_some(ViewOverrides {
            mode: ViewMode::Lit,
            show: self.args.show,
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
        self.args
            .graphics_validation
            .unwrap_or(cfg!(debug_assertions))
    }

    /// The master quality preset: the request (never persisted), else the
    /// `stored` choice. A first launch, with neither, seeds `Auto` into the
    /// store published in `resources` so it persists once.
    pub fn resolve_quality_preset(
        &self,
        stored: &StoredSettings,
        resources: &mut Resources,
    ) -> QualityPreset {
        if let Some(preset) = self
            .args
            .graphics_quality
            .or(stored.graphics.quality_preset)
        {
            return preset;
        }
        let mut seeded = stored.clone();
        seeded.graphics.quality_preset = Some(QualityPreset::Auto);
        seeded.store_changed(resources);
        QualityPreset::Auto
    }

    /// How the acceleration structure tracks moving props: the request, else `Auto`.
    pub fn resolve_rt_dynamic(&self) -> RtDynamicMode {
        self.args.rt_dynamic.unwrap_or_default()
    }

    /// The settings this launch overrides for its own run, layered over the
    /// user's persisted choices and never persisted: vsync, the frame-rate cap,
    /// the window mode, temporal upscaling, ray-traced reflections and the DLSS
    /// preset.
    pub fn session_overrides(&self) -> UserGraphicsSettings {
        UserGraphicsSettings {
            vsync: self.args.vsync,
            fps_cap: self.args.limit_fps,
            window_mode: self.args.window_mode,
            temporal_upscaling: self.args.upscaling,
            ray_traced_reflections: self.args.rt_reflections,
            dlss_preset: self.args.dlss_preset,
            ..UserGraphicsSettings::default()
        }
    }

    /// Whether skinned meshes join the acceleration structure: the request, else in.
    pub fn resolve_rt_skinned_geometry(&self) -> bool {
        self.args.rt_skinned_geometry.unwrap_or(true)
    }

    /// Whether the presented frame stays readable: in a dev session.
    pub fn frame_capture(&self) -> bool {
        self.dev_loop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::output::dlss::DlssPreset;

    // The world's own arguments win, and the published request carries them.
    #[test]
    fn publishing_adopts_the_worlds_launch_args() {
        let mut world = World::new();
        world.insert_resource(LaunchArgs {
            limit_frames: Some(3),
            ..Default::default()
        });
        let mut request = LaunchRequest::default();
        request.publish(&mut world);
        assert_eq!(request.args.limit_frames, Some(3));
        assert_eq!(world.resource::<LaunchRequest>(), Some(&request));
        assert!(world.resource::<LaunchArgs>().is_none());
        assert!(world.resource::<ViewOverrides>().is_none());
    }

    // Only `bench` arms a benchmark, stopped where the launch stops it.
    #[test]
    fn publishing_a_bench_arms_the_benchmark() {
        let mut world = World::new();
        LaunchRequest::default().publish(&mut world);
        assert!(world.resource::<crate::benchmark::BenchmarkRun>().is_none());

        let mut world = World::new();
        let mut request = LaunchRequest {
            args: LaunchArgs {
                command: crate::launch::LaunchCommand::Bench,
                stop_at_millis: Some(2_500),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        request.publish(&mut world);
        let run = world
            .resource::<crate::benchmark::BenchmarkRun>()
            .expect("armed");
        assert!(run.stop_seconds() <= 2.5);
    }

    // `--no-hud` starts the stats HUD hidden; without it nothing is published.
    #[test]
    fn publishing_no_hud_hides_the_stats_hud() {
        let mut world = World::new();
        LaunchRequest::default().publish(&mut world);
        assert!(world.resource::<crate::ecs::HudHidden>().is_none());

        let mut world = World::new();
        let mut request = LaunchRequest {
            args: LaunchArgs {
                hud: Some(false),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        request.publish(&mut world);
        assert_eq!(
            world.resource::<crate::ecs::HudHidden>(),
            Some(&crate::ecs::HudHidden(true))
        );
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
        let request = |graphics_validation| LaunchRequest {
            args: LaunchArgs {
                graphics_validation,
                ..LaunchArgs::NONE
            },
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
                graphics_quality: Some(QualityPreset::Ultra),
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
        for mode in RtDynamicMode::ALL {
            let request = LaunchRequest {
                args: LaunchArgs {
                    rt_dynamic: Some(mode),
                    ..LaunchArgs::NONE
                },
                ..Default::default()
            };
            assert_eq!(request.resolve_rt_dynamic(), mode);
        }
    }

    #[test]
    fn vsync_and_a_frame_rate_cap_are_session_overrides() {
        let forced = LaunchRequest {
            args: LaunchArgs {
                vsync: Some(true),
                limit_fps: Some(30),
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
    fn requested_upscaling_and_ray_tracing_are_session_overrides() {
        let user = UserGraphicsSettings {
            temporal_upscaling: Some(true),
            ray_traced_reflections: Some(true),
            ..Default::default()
        };
        let forced = |state| LaunchRequest {
            args: LaunchArgs {
                upscaling: state,
                rt_reflections: state,
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        let merged = |state| user.overlaid_by(&forced(state).session_overrides());
        assert_eq!(merged(Some(false)).temporal_upscaling, Some(false));
        assert_eq!(merged(Some(false)).ray_traced_reflections, Some(false));
        assert_eq!(merged(None).temporal_upscaling, Some(true));
        assert_eq!(merged(None).ray_traced_reflections, Some(true));
    }

    #[test]
    fn a_requested_dlss_preset_is_a_session_override() {
        assert_eq!(
            LaunchRequest::default().session_overrides(),
            UserGraphicsSettings::default()
        );
        let forced = LaunchRequest {
            args: LaunchArgs {
                dlss_preset: Some(DlssPreset::M),
                ..LaunchArgs::NONE
            },
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
            args: LaunchArgs {
                rt_skinned_geometry,
                ..LaunchArgs::NONE
            },
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
            args: LaunchArgs {
                show: ShowFlags::all().toggled(ShowFlags::REACTIVE),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        };
        let view = hidden.view_overrides().expect("view state");
        assert_eq!(view.mode, ViewMode::Lit);
        assert!(!view.show.contains(ShowFlags::REACTIVE));
        assert!(view.show.contains(ShowFlags::FOG));
    }

    #[test]
    fn frame_capture_is_armed_by_a_dev_loop() {
        assert!(!LaunchRequest::default().frame_capture());
        let dev_loop = LaunchRequest {
            dev_loop: true,
            ..Default::default()
        };
        assert!(dev_loop.frame_capture());
    }
}

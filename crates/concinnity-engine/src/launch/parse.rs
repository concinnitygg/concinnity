// Reading a launch from its command line: a command, then the flags it takes.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{
    LaunchArgs, LaunchArgsError, LaunchCommand, LaunchFlag, ScreenshotAt, flag, parse_seconds,
    run_millis, short_flag,
};
use crate::gfx::view_modes::ShowFlags;
use crate::platform::Platform;
use crate::render::output::dlss::DlssPreset;
use crate::render::rt_accel::geom::RtDynamicMode;
use crate::settings::QualityPreset;
use crate::window::WindowMode;

// The flag `name` names, without its `--`, and the state it sets: `false` for
// the `no-` form of a negatable switch.
fn named(name: &str) -> Option<(&'static LaunchFlag, bool)> {
    if let Some(flag) = flag(name) {
        return Some((flag, true));
    }
    let flag = flag(name.strip_prefix("no-")?).filter(|flag| flag.negatable)?;
    Some((flag, false))
}

// Set a negatable switch's state, refusing the opposite of one already given.
fn toggle(state: &mut Option<bool>, on: bool, name: &'static str) -> Result<(), LaunchArgsError> {
    if state.is_some_and(|was| was != on) {
        return Err(LaunchArgsError::Conflict(name));
    }
    *state = Some(on);
    Ok(())
}

impl LaunchArgs {
    /// The launch `args` ask for: an optional command from [`LaunchCommand`]
    /// (`run` when none is given), and flags from [`LAUNCH_FLAGS`] as `--name`,
    /// `--no-name`, `-q`, `--name value` or `--name=value`. `--help` is answered
    /// with [`LaunchArgsError::Help`]. Takes `std::env::args().skip(1)` as it is.
    ///
    /// # Errors
    ///
    /// Returns [`LaunchArgsError::Help`] for `--help`, or an error naming an
    /// unknown argument, a flag the command does not take, or a missing,
    /// unexpected, conflicting or malformed value.
    ///
    /// [`LAUNCH_FLAGS`]: super::LAUNCH_FLAGS
    pub fn parse<I>(args: I) -> Result<Self, LaunchArgsError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut parsed = Self::NONE;
        let mut command = None;
        let mut given: Vec<&'static str> = Vec::new();
        let mut args = args.into_iter().peekable();
        while let Some(item) = args.next() {
            let arg = item.as_ref();
            if arg == "--help" || arg == "-h" {
                return Err(LaunchArgsError::Help);
            }
            let unknown = || LaunchArgsError::Unknown(arg.to_string());
            let (flag, on, inline) = if let Some(long) = arg.strip_prefix("--") {
                let (name, inline) = match long.split_once('=') {
                    Some((name, value)) => (name, Some(value)),
                    None => (long, None),
                };
                let (flag, on) = named(name).ok_or_else(unknown)?;
                (flag, on, inline)
            } else if let Some(short) = arg.strip_prefix('-') {
                let mut chars = short.chars();
                let (Some(short), None) = (chars.next(), chars.next()) else {
                    return Err(unknown());
                };
                (short_flag(short).ok_or_else(unknown)?, true, None)
            } else {
                if command.is_some() {
                    return Err(unknown());
                }
                command = Some(LaunchCommand::from_name(arg).ok_or_else(unknown)?);
                continue;
            };
            let value: Option<String> = match (flag.value, inline) {
                (None, None) => None,
                (None, Some(_)) => return Err(LaunchArgsError::UnexpectedValue(flag.name)),
                (Some(_), Some(value)) => Some(value.to_string()),
                (Some(_), None) => {
                    let follows = args.peek().map(AsRef::as_ref).is_some_and(|next| {
                        flag.missing_value.is_none()
                            || !(next.starts_with('-') || LaunchCommand::from_name(next).is_some())
                    });
                    match (follows, flag.missing_value) {
                        (true, _) => args.next().map(|next| next.as_ref().to_string()),
                        (false, Some(missing)) => Some(missing.to_string()),
                        (false, None) => return Err(LaunchArgsError::MissingValue(flag.name)),
                    }
                }
            };
            let name = if on {
                flag.name.to_string()
            } else {
                alloc::format!("no-{}", flag.name)
            };
            parsed.set(&name, value.as_deref())?;
            given.push(flag.name);
        }
        parsed.command = command.unwrap_or_default();
        if let Some(flag) = given
            .into_iter()
            .find(|&name| flag(name).is_some_and(|flag| !flag.takes(parsed.command)))
        {
            return Err(LaunchArgsError::NotFor {
                flag,
                command: parsed.command,
            });
        }
        Ok(parsed)
    }

    /// Apply the flag `name` (without its `--`, and `no-name` for the off form
    /// of a negatable switch), with its value when it takes one.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is unknown, or `value` is missing,
    /// unexpected, conflicting or malformed.
    pub fn set(&mut self, name: &str, value: Option<&str>) -> Result<(), LaunchArgsError> {
        let (flag, on) = named(name).ok_or_else(|| LaunchArgsError::Unknown(name.to_string()))?;
        let bad = |value: &str| LaunchArgsError::BadValue {
            flag: flag.name,
            value: value.to_string(),
        };
        let seconds = |value: &str| {
            parse_seconds(value)
                .map(run_millis)
                .ok_or_else(|| bad(value))
        };
        match (flag.name, value) {
            ("quiet", None) => self.quiet = true,
            ("graphics-validation", None) => {
                toggle(&mut self.graphics_validation, on, flag.name)?;
            }
            ("upscaling", None) => toggle(&mut self.upscaling, on, flag.name)?,
            ("rt-reflections", None) => toggle(&mut self.rt_reflections, on, flag.name)?,
            ("rt-skinned-geometry", None) => {
                toggle(&mut self.rt_skinned_geometry, on, flag.name)?;
            }
            ("vsync", None) => toggle(&mut self.vsync, on, flag.name)?,
            ("hud", None) => toggle(&mut self.hud, on, flag.name)?,
            ("graphics-quality", Some(value)) => {
                self.graphics_quality = Some(
                    QualityPreset::ALL
                        .into_iter()
                        .find(|preset| preset.name().eq_ignore_ascii_case(value))
                        .ok_or_else(|| bad(value))?,
                );
            }
            ("graphics-api", Some(value)) => {
                self.graphics_api = Some(Platform::from_key(value).ok_or_else(|| bad(value))?);
            }
            ("window-mode", Some(value)) => {
                self.window_mode = Some(
                    WindowMode::ALL
                        .iter()
                        .copied()
                        .find(|mode| mode.as_str() == value)
                        .ok_or_else(|| bad(value))?,
                );
            }
            ("rt-dynamic", Some(value)) => {
                self.rt_dynamic = Some(
                    RtDynamicMode::ALL
                        .into_iter()
                        .find(|mode| mode.key() == value)
                        .ok_or_else(|| bad(value))?,
                );
            }
            ("dlss-preset", Some(value)) => {
                self.dlss_preset = Some(
                    DlssPreset::ALL
                        .into_iter()
                        .find(|preset| preset.label().eq_ignore_ascii_case(value))
                        .ok_or_else(|| bad(value))?,
                );
            }
            ("fixed-frame-rate", Some(value)) => {
                self.fixed_frame_rate = Some(value.parse().map_err(|_| bad(value))?);
            }
            ("screenshots-dir", Some(value)) => {
                if value.is_empty() {
                    return Err(bad(value));
                }
                self.screenshots_dir = Some(value.to_string());
            }
            ("hide-pass", Some(value)) => {
                let (pass, _) = ShowFlags::NAMED
                    .into_iter()
                    .find(|&(_, name)| name == value)
                    .ok_or_else(|| bad(value))?;
                self.show = self.show.with(pass, false);
            }
            ("limit-frames", Some(value)) => {
                self.limit_frames = Some(value.parse().map_err(|_| bad(value))?);
            }
            ("limit-fps", Some(value)) => {
                self.limit_fps = Some(value.parse().map_err(|_| bad(value))?);
            }
            ("debug-port", Some(value)) => {
                self.debug_port = Some(value.parse().map_err(|_| bad(value))?);
            }
            ("start-at", Some(value)) => self.start_at_millis = Some(seconds(value)?),
            ("stop-at", Some(value)) => self.stop_at_millis = Some(seconds(value)?),
            ("screenshot-at", Some(value)) => self
                .screenshot_at
                .push(ScreenshotAt::parse(value).ok_or_else(|| bad(value))?),
            (_, None) => return Err(LaunchArgsError::MissingValue(flag.name)),
            (_, Some(_)) => return Err(LaunchArgsError::UnexpectedValue(flag.name)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::{GRAPHICS_API_VALUES, LAUNCH_FLAGS};
    use alloc::string::String;
    use core::num::NonZeroU32;

    fn parse(args: &[&str]) -> Result<LaunchArgs, LaunchArgsError> {
        LaunchArgs::parse(args.iter().copied())
    }

    #[test]
    fn no_args_run_and_ask_for_nothing() {
        assert_eq!(parse(&[]), Ok(LaunchArgs::NONE));
        assert_eq!(LaunchArgs::NONE, LaunchArgs::default());
        assert_eq!(LaunchArgs::NONE.command, LaunchCommand::Run);
    }

    #[test]
    fn a_command_is_named_once_anywhere_among_the_flags() {
        assert_eq!(parse(&["bench"]).unwrap().command, LaunchCommand::Bench);
        assert_eq!(parse(&["cook"]).unwrap().command, LaunchCommand::Cook);
        let args = parse(&["--vsync", "bench", "--stop-at", "3"]).unwrap();
        assert_eq!(args.command, LaunchCommand::Bench);
        assert_eq!(args.vsync, Some(true));
        assert!(args.benchmarks());
        assert_eq!(
            parse(&["run", "bench"]),
            Err(LaunchArgsError::Unknown("bench".into()))
        );
        assert_eq!(
            parse(&["debug"]),
            Err(LaunchArgsError::Unknown("debug".into()))
        );
    }

    // Moments a millisecond apart get files of their own.
    #[test]
    fn a_screenshot_at_a_moment_is_named_for_it_to_the_millisecond() {
        let args = parse(&["bench", "--screenshot-at=1.001", "--screenshot-at=1.004"]).unwrap();
        let files: Vec<&str> = args.screenshot_at.iter().map(|s| s.file.as_str()).collect();
        assert_eq!(files, ["at-1.001s.png", "at-1.004s.png"]);
    }

    #[test]
    fn every_flag_parses() {
        let args = parse(&[
            "bench",
            "-q",
            "--vsync",
            "--no-hud",
            "--limit-fps",
            "144",
            "--graphics-quality=ultra",
            "--graphics-api",
            "vulkan",
            "--window-mode=fullscreen",
            "--no-graphics-validation",
            "--no-upscaling",
            "--rt-dynamic",
            "rebuild",
            "--rt-reflections",
            "--no-rt-skinned-geometry",
            "--dlss-preset",
            "M",
            "--fixed-frame-rate",
            "60",
            "--screenshots-dir=shots/vk",
            "--hide-pass",
            "fog",
            "--hide-pass=reactive",
            "--limit-frames",
            "600",
            "--debug-port",
            "9001",
            "--start-at=96.5",
            "--stop-at",
            "120",
            "--screenshot-at",
            "12.5:beach",
            "--screenshot-at=30",
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                command: LaunchCommand::Bench,
                quiet: true,
                graphics_quality: Some(QualityPreset::Ultra),
                graphics_api: Some(Platform::Vulkan),
                window_mode: Some(WindowMode::Fullscreen),
                graphics_validation: Some(false),
                upscaling: Some(false),
                rt_dynamic: Some(RtDynamicMode::Rebuild),
                rt_reflections: Some(true),
                rt_skinned_geometry: Some(false),
                vsync: Some(true),
                hud: Some(false),
                dlss_preset: Some(DlssPreset::M),
                fixed_frame_rate: NonZeroU32::new(60),
                screenshots_dir: Some("shots/vk".into()),
                show: ShowFlags::all()
                    .with(ShowFlags::FOG, false)
                    .with(ShowFlags::REACTIVE, false),
                limit_frames: Some(600),
                limit_fps: Some(144),
                debug_port: Some(9001),
                start_at_millis: Some(96_500),
                stop_at_millis: Some(120_000),
                screenshot_at: alloc::vec![
                    ScreenshotAt::new(12.5, "beach.png"),
                    ScreenshotAt::new(30.0, "at-30.000s.png"),
                ],
            }
        );
    }

    #[test]
    fn every_flag_in_the_table_is_handled() {
        for flag in LAUNCH_FLAGS {
            let value = flag
                .values
                .first()
                .copied()
                .or_else(|| flag.value.map(|_| "1"));
            let mut args = LaunchArgs::NONE;
            assert_eq!(args.set(flag.name, value), Ok(()), "{}", flag.name);
            assert_ne!(args, LaunchArgs::NONE, "{}", flag.name);
            if flag.negatable {
                let mut off = LaunchArgs::NONE;
                let name = alloc::format!("no-{}", flag.name);
                assert_eq!(off.set(&name, None), Ok(()), "{name}");
                assert_ne!(off, args, "{name}");
            }
        }
    }

    #[test]
    fn a_switch_and_its_negation_exclude_each_other() {
        assert_eq!(parse(&["--no-vsync"]).unwrap().vsync, Some(false));
        assert_eq!(parse(&["--no-hud"]).unwrap().hud, Some(false));
        assert_eq!(parse(&["--vsync", "--vsync"]).unwrap().vsync, Some(true));
        assert_eq!(
            parse(&["--vsync", "--no-vsync"]),
            Err(LaunchArgsError::Conflict("vsync"))
        );
        assert_eq!(
            parse(&["--no-quiet"]),
            Err(LaunchArgsError::Unknown("--no-quiet".into()))
        );
    }

    #[test]
    fn a_bare_debug_port_takes_the_default() {
        let port = |args: &[&str]| parse(args).unwrap().debug_port;
        assert_eq!(port(&["--debug-port"]), Some(8777));
        assert_eq!(port(&["--debug-port", "--vsync"]), Some(8777));
        assert_eq!(port(&["--debug-port", "bench"]), Some(8777));
        assert_eq!(port(&["--debug-port", "9001"]), Some(9001));
    }

    #[test]
    fn a_flag_outside_its_command_is_refused() {
        for args in [&["--stop-at", "3"][..], &["run", "--start-at", "3"]] {
            assert_eq!(
                parse(args),
                Err(LaunchArgsError::NotFor {
                    flag: args[args.len() - 2].trim_start_matches('-'),
                    command: LaunchCommand::Run,
                }),
                "{args:?}"
            );
        }
        assert_eq!(
            parse(&["cook", "--vsync"]),
            Err(LaunchArgsError::NotFor {
                flag: "vsync",
                command: LaunchCommand::Cook,
            })
        );
        assert!(parse(&["cook", "-q"]).unwrap().quiet);
    }

    #[test]
    fn mistakes_name_what_went_wrong() {
        assert_eq!(
            parse(&["--fast"]),
            Err(LaunchArgsError::Unknown("--fast".into()))
        );
        assert_eq!(parse(&["-x"]), Err(LaunchArgsError::Unknown("-x".into())));
        assert_eq!(parse(&["-qq"]), Err(LaunchArgsError::Unknown("-qq".into())));
        assert_eq!(
            parse(&["--limit-frames"]),
            Err(LaunchArgsError::MissingValue("limit-frames"))
        );
        assert_eq!(
            parse(&["--vsync=on"]),
            Err(LaunchArgsError::UnexpectedValue("vsync"))
        );
        for (flag, value) in [
            ("limit-fps", "fast"),
            ("graphics-quality", "max"),
            ("graphics-api", "opengl"),
            ("window-mode", "maximized"),
            ("rt-dynamic", "sometimes"),
            ("dlss-preset", "z"),
            ("fixed-frame-rate", "0"),
            ("hide-pass", "everything"),
            ("debug-port", "99999"),
        ] {
            assert_eq!(
                parse(&[&alloc::format!("--{flag}"), value]),
                Err(LaunchArgsError::BadValue {
                    flag,
                    value: value.into()
                }),
                "{flag}"
            );
        }
        assert_eq!(
            parse(&["--screenshots-dir="]),
            Err(LaunchArgsError::BadValue {
                flag: "screenshots-dir",
                value: String::new()
            })
        );
        for value in [
            "soon",
            "-1",
            "3:../out",
            "3:/tmp/out.png",
            "3:",
            "3:shots/",
            "3:a.png:stream",
        ] {
            assert_eq!(
                parse(&["bench", "--screenshot-at", value]),
                Err(LaunchArgsError::BadValue {
                    flag: "screenshot-at",
                    value: value.into()
                }),
                "{value}"
            );
        }
        for flag in ["start-at", "stop-at"] {
            for value in ["soon", "-1", "inf"] {
                assert_eq!(
                    parse(&["bench", &alloc::format!("--{flag}"), value]),
                    Err(LaunchArgsError::BadValue {
                        flag,
                        value: value.into()
                    }),
                    "{flag} {value}"
                );
            }
        }
        let args = parse(&["bench", "--start-at=0.25", "--stop-at=2"]).unwrap();
        assert_eq!(args.start_at(), Some(0.25));
        assert_eq!(args.stop_at(), Some(2.0));
        assert_eq!(LaunchArgs::NONE.start_at(), None);
        assert_eq!(LaunchArgs::NONE.stop_at(), None);
    }

    // The flag names every platform by its key, and only those.
    #[test]
    fn every_graphics_api_is_named_by_its_key() {
        let keys: Vec<&str> = Platform::ALL.iter().map(|p| p.key()).collect();
        assert_eq!(keys, GRAPHICS_API_VALUES);
        for platform in Platform::ALL {
            let args = parse(&["--graphics-api", platform.key()]).unwrap();
            assert_eq!(args.graphics_api, Some(platform));
        }
    }

    #[test]
    fn owned_arguments_parse_as_they_come() {
        let argv = alloc::vec![String::from("--limit-fps"), String::from("60")];
        assert_eq!(LaunchArgs::parse(argv).map(|a| a.limit_fps), Ok(Some(60)));
    }

    #[test]
    fn help_is_its_own_answer() {
        assert_eq!(parse(&["--vsync", "--help"]), Err(LaunchArgsError::Help));
        assert_eq!(parse(&["bench", "-h"]), Err(LaunchArgsError::Help));
    }
}

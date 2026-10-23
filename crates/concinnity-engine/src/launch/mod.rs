//! What a launch asks of the run, from the command line or from code: the
//! [`LaunchArgs`] parsed from the flags in [`LAUNCH_FLAGS`].
//!
//! The flags are described once, in the table, so every front that offers them
//! (an app's own command line, `cn run`, a debug client) shows the same names
//! and descriptions.

mod error;
mod flags;
mod pipeline;
mod request;

pub use error::LaunchArgsError;
pub use flags::{
    GRAPHICS_API_VALUES, LAUNCH_FLAGS, LaunchFlag, QUALITY_PRESET_VALUES, flag, usage,
};
pub use pipeline::PipelineMode;
pub use request::{DEFAULT_SCREENSHOTS_DIR, LaunchRequest, screenshot_path};

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::platform::Platform;
use crate::settings::QualityPreset;
use concinnity_core::math::{SaturatingFrom, round};

/// What a launch asks of the run. Each field left at its default leaves the
/// run as the world and the saved settings make it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaunchArgs {
    /// Benchmark the world: arm its `BenchmarkConfig`, or the default one, and
    /// report what the frames cost when its last phase ends.
    pub benchmark: bool,
    /// Turn vertical sync on, over the world's settings and the saved ones.
    pub vsync: bool,
    /// Cap the frame rate at this many frames a second, `0` for uncapped,
    /// over the world's settings, the saved ones and a benchmark's own cap.
    pub fps_cap: Option<u32>,
    /// Force the graphics-quality preset for this run, unsaved.
    pub quality_preset: Option<QualityPreset>,
    /// Render with this graphics API, over the one chosen in the saved
    /// settings, without saving it.
    pub graphics_api: Option<Platform>,
    /// The directory screenshots are written under, instead of
    /// [`DEFAULT_SCREENSHOTS_DIR`].
    pub screenshots_dir: Option<String>,
    /// Stop after this many frames.
    pub frames: Option<u64>,
    /// Take the world's authored screenshots, as a benchmarked run always
    /// does. A run that asks for neither takes none of them.
    pub screenshots: bool,
    /// Screenshots to take at moments of the run, with or without the
    /// world's own. A file another shot writes is taken only once.
    pub screenshot_at: Vec<ScreenshotAt>,
}

/// A screenshot a launch asks for at a moment of the run, as
/// `--screenshot-at SECONDS[:FILE]` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenshotAt {
    // Milliseconds of the run, so the launch stays comparable for equality.
    millis: u32,
    /// The image's path under the screenshot directory.
    pub file: String,
}

impl ScreenshotAt {
    /// A screenshot `seconds` into the run, saved as `file`.
    pub fn new(seconds: f32, file: &str) -> Self {
        Self {
            millis: u32::saturating_from(round(seconds.max(0.0) * 1000.0)),
            file: file.to_string(),
        }
    }

    /// Seconds into the run, on the simulation clock.
    pub fn seconds(&self) -> f32 {
        self.millis as f32 / 1000.0
    }

    // `SECONDS[:FILE]`, the file named for the moment when left out and given
    // a `.png` extension when it has none. `None` for a moment that is not a
    // number or a file that would leave the screenshot directory.
    fn parse(value: &str) -> Option<Self> {
        let (seconds, file) = match value.split_once(':') {
            Some((seconds, file)) => (seconds, Some(file)),
            None => (value, None),
        };
        let seconds: f32 = seconds
            .parse()
            .ok()
            .filter(|s: &f32| s.is_finite() && *s >= 0.0)?;
        let mut shot = Self::new(seconds, "");
        shot.file = match file {
            Some(file) if !crate::screenshot::is_contained_file(file) => return None,
            Some(file)
                if file
                    .rsplit(['/', '\\'])
                    .next()
                    .is_some_and(|f| f.contains('.')) =>
            {
                file.to_string()
            }
            Some(file) => alloc::format!("{file}.png"),
            None => alloc::format!("at-{}.{:03}s.png", shot.millis / 1000, shot.millis % 1000),
        };
        Some(shot)
    }
}

impl LaunchArgs {
    /// A launch that asks for nothing.
    pub const NONE: Self = Self {
        benchmark: false,
        vsync: false,
        fps_cap: None,
        quality_preset: None,
        graphics_api: None,
        screenshots_dir: None,
        frames: None,
        screenshots: false,
        screenshot_at: Vec::new(),
    };

    /// The launch `args` ask for, each a flag from [`LAUNCH_FLAGS`] as
    /// `--name`, `--name value` or `--name=value`. `--help` is answered with
    /// [`LaunchArgsError::Help`]. Takes `std::env::args().skip(1)` as it is.
    ///
    /// # Errors
    ///
    /// Returns [`LaunchArgsError::Help`] for `--help`, or an error naming an
    /// unknown flag or a missing, unexpected, or malformed value.
    pub fn parse<I>(args: I) -> Result<Self, LaunchArgsError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut parsed = Self::NONE;
        let mut args = args.into_iter();
        while let Some(item) = args.next() {
            let arg = item.as_ref();
            if arg == "--help" || arg == "-h" {
                return Err(LaunchArgsError::Help);
            }
            let Some(named) = arg.strip_prefix("--") else {
                return Err(LaunchArgsError::Unknown(arg.to_string()));
            };
            let (name, inline) = match named.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (named, None),
            };
            let Some(flag) = flag(name) else {
                return Err(LaunchArgsError::Unknown(arg.to_string()));
            };
            let following;
            let value = match (flag.value, inline) {
                (None, None) => None,
                (None, Some(_)) => return Err(LaunchArgsError::UnexpectedValue(flag.name)),
                (Some(_), Some(value)) => Some(value),
                (Some(_), None) => {
                    following = args
                        .next()
                        .ok_or(LaunchArgsError::MissingValue(flag.name))?;
                    Some(following.as_ref())
                }
            };
            parsed.set(flag.name, value)?;
        }
        Ok(parsed)
    }

    /// Apply the flag `name` (without its `--`), with its value when it takes
    /// one.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is unknown, or `value` is missing,
    /// unexpected, or malformed.
    pub fn set(&mut self, name: &str, value: Option<&str>) -> Result<(), LaunchArgsError> {
        let flag = flag(name).ok_or_else(|| LaunchArgsError::Unknown(name.to_string()))?;
        let bad = |value: &str| LaunchArgsError::BadValue {
            flag: flag.name,
            value: value.to_string(),
        };
        match (flag.name, value) {
            ("benchmark", None) => self.benchmark = true,
            ("vsync", None) => self.vsync = true,
            ("screenshots", None) => self.screenshots = true,
            ("screenshot-at", Some(value)) => self
                .screenshot_at
                .push(ScreenshotAt::parse(value).ok_or_else(|| bad(value))?),
            ("fps-cap", Some(value)) => self.fps_cap = Some(value.parse().map_err(|_| bad(value))?),
            ("frames", Some(value)) => self.frames = Some(value.parse().map_err(|_| bad(value))?),
            ("quality-preset", Some(value)) => {
                self.quality_preset = Some(
                    QualityPreset::ALL
                        .into_iter()
                        .find(|preset| preset.name().eq_ignore_ascii_case(value))
                        .ok_or_else(|| bad(value))?,
                );
            }
            ("screenshots-dir", Some(value)) => {
                if value.is_empty() {
                    return Err(bad(value));
                }
                self.screenshots_dir = Some(value.to_string());
            }
            ("graphics-api", Some(value)) => {
                self.graphics_api = Some(Platform::from_key(value).ok_or_else(|| bad(value))?);
            }
            (_, None) => return Err(LaunchArgsError::MissingValue(flag.name)),
            (_, Some(_)) => return Err(LaunchArgsError::UnexpectedValue(flag.name)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    fn parse(args: &[&str]) -> Result<LaunchArgs, LaunchArgsError> {
        LaunchArgs::parse(args.iter().copied())
    }

    #[test]
    fn no_args_ask_for_nothing() {
        assert_eq!(parse(&[]), Ok(LaunchArgs::NONE));
        assert_eq!(LaunchArgs::NONE, LaunchArgs::default());
    }

    // Moments a millisecond apart get files of their own.
    #[test]
    fn a_screenshot_at_a_moment_is_named_for_it_to_the_millisecond() {
        let args = parse(&["--screenshot-at=1.001", "--screenshot-at=1.004"]).unwrap();
        let files: Vec<&str> = args.screenshot_at.iter().map(|s| s.file.as_str()).collect();
        assert_eq!(files, ["at-1.001s.png", "at-1.004s.png"]);
    }

    #[test]
    fn every_flag_parses() {
        let args = parse(&[
            "--benchmark",
            "--vsync",
            "--fps-cap",
            "144",
            "--quality-preset=ultra",
            "--graphics-api",
            "vulkan",
            "--screenshots-dir=shots/vk",
            "--frames",
            "600",
            "--screenshots",
            "--screenshot-at",
            "12.5:beach",
            "--screenshot-at=30",
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                benchmark: true,
                vsync: true,
                fps_cap: Some(144),
                quality_preset: Some(QualityPreset::Ultra),
                graphics_api: Some(Platform::Vulkan),
                screenshots_dir: Some("shots/vk".into()),
                frames: Some(600),
                screenshots: true,
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
        }
    }

    #[test]
    fn mistakes_name_what_went_wrong() {
        assert_eq!(
            parse(&["--fast"]),
            Err(LaunchArgsError::Unknown("--fast".into()))
        );
        assert_eq!(
            parse(&["benchmark"]),
            Err(LaunchArgsError::Unknown("benchmark".into()))
        );
        assert_eq!(
            parse(&["--frames"]),
            Err(LaunchArgsError::MissingValue("frames"))
        );
        assert_eq!(
            parse(&["--vsync=on"]),
            Err(LaunchArgsError::UnexpectedValue("vsync"))
        );
        assert_eq!(
            parse(&["--fps-cap", "fast"]),
            Err(LaunchArgsError::BadValue {
                flag: "fps-cap",
                value: "fast".into()
            })
        );
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
                parse(&["--screenshot-at", value]),
                Err(LaunchArgsError::BadValue {
                    flag: "screenshot-at",
                    value: value.into()
                }),
                "{value}"
            );
        }
        assert_eq!(
            parse(&["--graphics-api", "opengl"]),
            Err(LaunchArgsError::BadValue {
                flag: "graphics-api",
                value: "opengl".into()
            })
        );
        assert_eq!(
            parse(&["--quality-preset", "max"]),
            Err(LaunchArgsError::BadValue {
                flag: "quality-preset",
                value: "max".into()
            })
        );
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
        let argv = alloc::vec![String::from("--fps-cap"), String::from("60")];
        assert_eq!(LaunchArgs::parse(argv).map(|a| a.fps_cap), Ok(Some(60)));
    }

    #[test]
    fn help_is_its_own_answer() {
        assert_eq!(parse(&["--vsync", "--help"]), Err(LaunchArgsError::Help));
    }
}

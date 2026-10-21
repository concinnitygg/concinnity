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
pub use flags::{LAUNCH_FLAGS, LaunchFlag, QUALITY_PRESET_VALUES, flag, usage};
pub use pipeline::PipelineMode;
pub use request::LaunchRequest;

use alloc::string::ToString;

use crate::settings::QualityPreset;

/// What a launch asks of the run. Each field left at its default leaves the
/// run as the world and the saved settings make it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// Stop after this many frames.
    pub frames: Option<u64>,
}

impl LaunchArgs {
    /// A launch that asks for nothing.
    pub const NONE: Self = Self {
        benchmark: false,
        vsync: false,
        fps_cap: None,
        quality_preset: None,
        frames: None,
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

    #[test]
    fn every_flag_parses() {
        let args = parse(&[
            "--benchmark",
            "--vsync",
            "--fps-cap",
            "144",
            "--quality-preset=ultra",
            "--frames",
            "600",
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                benchmark: true,
                vsync: true,
                fps_cap: Some(144),
                quality_preset: Some(QualityPreset::Ultra),
                frames: Some(600),
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
            parse(&["--quality-preset", "max"]),
            Err(LaunchArgsError::BadValue {
                flag: "quality-preset",
                value: "max".into()
            })
        );
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

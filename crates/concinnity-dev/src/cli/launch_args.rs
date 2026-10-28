// The launch flags on the command line, built from the table the engine
// describes them in, so `cn --help` and a shipped app's usage text agree.

use clap::builder::PossibleValuesParser;
use clap::{Arg, ArgAction, ArgMatches, Command};
use concinnity_engine::launch::{LAUNCH_FLAGS, LaunchArgs};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct LaunchArgsArg(pub(crate) LaunchArgs);

impl LaunchArgsArg {
    pub(crate) const NONE: Self = Self(LaunchArgs::NONE);
}

impl LaunchArgsArg {
    fn set(&mut self, name: &str, value: Option<&str>) -> Result<(), clap::Error> {
        self.0.set(name, value).map_err(|e| {
            let first = e.to_string().lines().next().unwrap_or_default().to_string();
            clap::Error::raw(clap::error::ErrorKind::ValueValidation, first + "\n")
        })
    }
}

impl clap::Args for LaunchArgsArg {
    fn augment_args(cmd: Command) -> Command {
        LAUNCH_FLAGS.iter().fold(cmd, |cmd, flag| {
            let arg = Arg::new(flag.name).long(flag.name).help(flag.description);
            let arg = if flag.repeatable {
                arg.action(ArgAction::Append)
            } else {
                arg
            };
            cmd.arg(match flag.value {
                None => arg.action(ArgAction::SetTrue),
                Some(value) if flag.values.is_empty() => arg.value_name(value),
                Some(value) => arg
                    .value_name(value)
                    .value_parser(PossibleValuesParser::new(flag.values)),
            })
        })
    }

    fn augment_args_for_update(cmd: Command) -> Command {
        Self::augment_args(cmd)
    }
}

impl clap::FromArgMatches for LaunchArgsArg {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        let mut args = Self::NONE;
        args.update_from_arg_matches(matches)?;
        Ok(args)
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        for flag in LAUNCH_FLAGS {
            match flag.value {
                None if matches.get_flag(flag.name) => self.set(flag.name, None)?,
                None => {}
                Some(_) => {
                    for value in matches.get_many::<String>(flag.name).into_iter().flatten() {
                        self.set(flag.name, Some(value))?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use concinnity_engine::settings::QualityPreset;

    #[derive(Debug, Parser)]
    struct Probe {
        #[command(flatten)]
        launch: LaunchArgsArg,
    }

    fn parse(args: &[&str]) -> Result<LaunchArgs, clap::Error> {
        Probe::try_parse_from(core::iter::once("probe").chain(args.iter().copied()))
            .map(|p| p.launch.0)
    }

    #[test]
    fn every_table_flag_is_a_command_line_flag_with_its_description() {
        let cmd = Probe::command();
        for flag in LAUNCH_FLAGS {
            let arg = cmd
                .get_arguments()
                .find(|a| a.get_long() == Some(flag.name))
                .unwrap_or_else(|| panic!("--{} is offered", flag.name));
            assert_eq!(
                arg.get_help().map(ToString::to_string).as_deref(),
                Some(flag.description)
            );
        }
    }

    #[test]
    fn flags_parse_into_the_same_args_the_engine_reads() {
        assert_eq!(parse(&[]).unwrap(), LaunchArgs::NONE);
        let args = parse(&[
            "--benchmark",
            "--vsync",
            "--fps-cap",
            "90",
            "--quality-preset",
            "ultra",
            "--frames",
            "5",
            "--graphics-api",
            "directx",
            "--window-mode",
            "borderless",
            "--upscaling",
            "on",
            "--rt-reflections",
            "off",
            "--screenshots-dir",
            "shots",
            "--screenshots",
            "--screenshot-at",
            "4:start",
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                benchmark: true,
                vsync: true,
                fps_cap: Some(90),
                quality_preset: Some(QualityPreset::Ultra),
                graphics_api: Some(concinnity_engine::platform::Platform::DirectX),
                window_mode: Some(concinnity_engine::window::WindowMode::Borderless),
                upscaling: Some(true),
                rt_reflections: Some(false),
                screenshots_dir: Some("shots".into()),
                frames: Some(5),
                start_at_millis: None,
                screenshots: true,
                screenshot_at: vec![concinnity_engine::launch::ScreenshotAt::new(
                    4.0,
                    "start.png"
                )],
            }
        );
    }

    #[test]
    fn a_repeatable_flag_keeps_every_value_in_order() {
        use concinnity_engine::launch::ScreenshotAt;
        let args = parse(&[
            "--screenshot-at",
            "17.5:a.png",
            "--screenshot-at",
            "34.5:b.png",
        ])
        .unwrap();
        assert_eq!(
            args.screenshot_at,
            vec![
                ScreenshotAt::new(17.5, "a.png"),
                ScreenshotAt::new(34.5, "b.png")
            ]
        );
    }

    #[test]
    fn a_single_value_flag_is_refused_twice() {
        assert!(parse(&["--frames", "5", "--frames", "6"]).is_err());
    }

    #[test]
    fn a_bad_value_is_refused_at_the_flag() {
        assert!(parse(&["--fps-cap", "fast"]).is_err());
        assert!(parse(&["--quality-preset", "max"]).is_err());
    }
}

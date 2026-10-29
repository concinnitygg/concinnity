// The launch flags on the command line, built from the table the engine
// describes them in, so `cn --help` and a shipped app's usage text agree.

use core::marker::PhantomData;

use clap::builder::PossibleValuesParser;
use clap::{Arg, ArgAction, ArgMatches, Command};
use concinnity_engine::launch::{LAUNCH_FLAGS, LaunchArgs, LaunchCommand, LaunchFlag};

// The launch command whose flags a `LaunchArgsArg` offers.
pub(crate) trait LaunchScope {
    const COMMAND: LaunchCommand;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RunScope;

impl LaunchScope for RunScope {
    const COMMAND: LaunchCommand = LaunchCommand::Run;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BenchScope;

impl LaunchScope for BenchScope {
    const COMMAND: LaunchCommand = LaunchCommand::Bench;
}

// The flags `S`'s command takes, as the launch they parse into. The flags every
// command takes are the root command's own, so they are left out here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchArgsArg<S> {
    pub(crate) args: LaunchArgs,
    scope: PhantomData<S>,
}

impl<S: LaunchScope> LaunchArgsArg<S> {
    pub(crate) fn none() -> Self {
        Self {
            args: LaunchArgs {
                command: S::COMMAND,
                ..LaunchArgs::NONE
            },
            scope: PhantomData,
        }
    }

    fn flags() -> impl Iterator<Item = &'static LaunchFlag> {
        LAUNCH_FLAGS.iter().filter(|flag| {
            flag.takes(S::COMMAND) && !LaunchCommand::ALL.into_iter().all(|c| flag.takes(c))
        })
    }

    fn set(&mut self, name: &str, value: Option<&str>) -> Result<(), clap::Error> {
        self.args.set(name, value).map_err(|e| {
            let first = e.to_string().lines().next().unwrap_or_default().to_string();
            clap::Error::raw(clap::error::ErrorKind::ValueValidation, first + "\n")
        })
    }
}

fn negated(flag: &LaunchFlag) -> String {
    format!("no-{}", flag.name)
}

impl<S: LaunchScope> clap::Args for LaunchArgsArg<S> {
    fn augment_args(cmd: Command) -> Command {
        Self::flags().fold(cmd, |cmd, flag| {
            let arg = Arg::new(flag.name).long(flag.name).help(flag.description);
            let arg = if flag.repeatable {
                arg.action(ArgAction::Append)
            } else {
                arg
            };
            match flag.value {
                None if flag.negatable => {
                    let off = Arg::new(negated(flag))
                        .long(negated(flag))
                        .help(format!("Turn off --{}", flag.name))
                        .action(ArgAction::SetTrue)
                        .conflicts_with(flag.name);
                    cmd.arg(arg.action(ArgAction::SetTrue)).arg(off)
                }
                None => cmd.arg(arg.action(ArgAction::SetTrue)),
                Some(value) => {
                    let arg = arg.value_name(value);
                    let arg = match flag.missing_value {
                        Some(missing) => arg.num_args(0..=1).default_missing_value(missing),
                        None => arg,
                    };
                    cmd.arg(if flag.values.is_empty() {
                        arg
                    } else {
                        arg.value_parser(PossibleValuesParser::new(flag.values))
                    })
                }
            }
        })
    }

    fn augment_args_for_update(cmd: Command) -> Command {
        Self::augment_args(cmd)
    }
}

impl<S: LaunchScope> clap::FromArgMatches for LaunchArgsArg<S> {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        let mut args = Self::none();
        args.update_from_arg_matches(matches)?;
        Ok(args)
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        for flag in Self::flags() {
            match flag.value {
                None => {
                    if matches.get_flag(flag.name) {
                        self.set(flag.name, None)?;
                    }
                    if flag.negatable && matches.get_flag(&negated(flag)) {
                        self.set(&negated(flag), None)?;
                    }
                }
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
    struct Probe<S: LaunchScope + Clone + Send + Sync + 'static> {
        #[command(flatten)]
        launch: LaunchArgsArg<S>,
    }

    fn parse<S>(args: &[&str]) -> Result<LaunchArgs, clap::Error>
    where
        S: LaunchScope + Clone + Send + Sync + 'static,
    {
        Probe::<S>::try_parse_from(core::iter::once("probe").chain(args.iter().copied()))
            .map(|p| p.launch.args)
    }

    #[test]
    fn every_scoped_table_flag_is_a_command_line_flag_with_its_description() {
        let cmd = Probe::<BenchScope>::command();
        for flag in LaunchArgsArg::<BenchScope>::flags() {
            let arg = cmd
                .get_arguments()
                .find(|a| a.get_long() == Some(flag.name))
                .unwrap_or_else(|| panic!("--{} is offered", flag.name));
            assert_eq!(
                arg.get_help().map(ToString::to_string).as_deref(),
                Some(flag.description)
            );
        }
        assert!(cmd.get_arguments().all(|a| a.get_long() != Some("quiet")));
    }

    #[test]
    fn run_offers_no_bench_flag() {
        let cmd = Probe::<RunScope>::command();
        for name in ["start-at", "stop-at", "screenshot-at"] {
            assert!(
                cmd.get_arguments().all(|a| a.get_long() != Some(name)),
                "{name}"
            );
        }
        assert!(parse::<RunScope>(&["--stop-at", "3"]).is_err());
    }

    #[test]
    fn flags_parse_into_the_same_args_the_engine_reads() {
        assert_eq!(
            parse::<RunScope>(&[]).unwrap(),
            LaunchArgsArg::<RunScope>::none().args
        );
        let args = parse::<BenchScope>(&[
            "--vsync",
            "--limit-fps",
            "90",
            "--graphics-quality",
            "ultra",
            "--no-upscaling",
            "--hide-pass",
            "fog",
            "--debug-port",
            "--stop-at",
            "5",
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                command: LaunchCommand::Bench,
                vsync: Some(true),
                limit_fps: Some(90),
                graphics_quality: Some(QualityPreset::Ultra),
                upscaling: Some(false),
                show: concinnity_engine::gfx::view_modes::ShowFlags::all()
                    .with(concinnity_engine::gfx::view_modes::ShowFlags::FOG, false),
                debug_port: Some(concinnity_engine::launch::DEFAULT_DEBUG_PORT),
                stop_at_millis: Some(5_000),
                ..LaunchArgs::NONE
            }
        );
    }

    #[test]
    fn a_switch_and_its_negation_conflict() {
        assert!(parse::<RunScope>(&["--vsync", "--no-vsync"]).is_err());
        assert_eq!(
            parse::<RunScope>(&["--no-graphics-validation"])
                .unwrap()
                .graphics_validation,
            Some(false)
        );
    }

    #[test]
    fn a_repeatable_flag_keeps_every_value_in_order() {
        use concinnity_engine::launch::ScreenshotAt;
        let args = parse::<BenchScope>(&[
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
        assert!(parse::<RunScope>(&["--limit-frames", "5", "--limit-frames", "6"]).is_err());
    }

    #[test]
    fn a_bad_value_is_refused_at_the_flag() {
        assert!(parse::<RunScope>(&["--limit-fps", "fast"]).is_err());
        assert!(parse::<RunScope>(&["--graphics-quality", "max"]).is_err());
        assert!(parse::<RunScope>(&["--fixed-frame-rate", "0"]).is_err());
    }
}

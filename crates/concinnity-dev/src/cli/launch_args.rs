// The launch flags on the command line, built from the table the engine
// describes them in, so `cn --help` and a shipped app's usage text agree.

use clap::builder::PossibleValuesParser;
use clap::{Arg, ArgAction, ArgMatches, Command};
use concinnity_core::launch::{LAUNCH_FLAGS, LaunchArgs};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaunchArgsArg(pub(crate) LaunchArgs);

impl LaunchArgsArg {
    pub(crate) const NONE: Self = Self(LaunchArgs::NONE);
}

impl clap::Args for LaunchArgsArg {
    fn augment_args(cmd: Command) -> Command {
        LAUNCH_FLAGS.iter().fold(cmd, |cmd, flag| {
            let arg = Arg::new(flag.name).long(flag.name).help(flag.description);
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
            let value = match flag.value {
                None if matches.get_flag(flag.name) => None,
                None => continue,
                Some(_) => match matches.get_one::<String>(flag.name) {
                    Some(value) => Some(value.as_str()),
                    None => continue,
                },
            };
            self.0.set(flag.name, value).map_err(|e| {
                let first = e.to_string().lines().next().unwrap_or_default().to_string();
                clap::Error::raw(clap::error::ErrorKind::ValueValidation, first + "\n")
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use concinnity_core::settings::QualityPreset;

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
        ])
        .unwrap();
        assert_eq!(
            args,
            LaunchArgs {
                benchmark: true,
                vsync: true,
                fps_cap: Some(90),
                quality_preset: Some(QualityPreset::Ultra),
                frames: Some(5),
            }
        );
    }

    #[test]
    fn a_bad_value_is_refused_at_the_flag() {
        assert!(parse(&["--fps-cap", "fast"]).is_err());
        assert!(parse(&["--quality-preset", "max"]).is_err());
    }
}

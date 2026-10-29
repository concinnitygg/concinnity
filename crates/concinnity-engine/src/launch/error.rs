// What can go wrong reading launch flags.

use alloc::string::String;
use core::fmt;

use crate::launch::{LaunchCommand, flag, usage};

/// Why a launch's flags could not be read. Each message names the flag at
/// fault and ends with the usage text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchArgsError {
    /// `--help` was asked for. Its message is the usage text alone.
    Help,
    /// An argument that names no flag.
    Unknown(String),
    /// A flag that takes a value was given none.
    MissingValue(&'static str),
    /// A switch was given a value.
    UnexpectedValue(&'static str),
    /// A switch was turned both on and off.
    Conflict(&'static str),
    /// A flag the command does not take.
    NotFor {
        /// The flag, without its `--`.
        flag: &'static str,
        /// The command it was given to.
        command: LaunchCommand,
    },
    /// A flag's value is not one it accepts.
    BadValue {
        /// The flag, without its `--`.
        flag: &'static str,
        /// What it was given.
        value: String,
    },
}

impl fmt::Display for LaunchArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Help => return f.write_str(&usage()),
            Self::Unknown(arg) => write!(f, "unknown argument {arg:?}")?,
            Self::MissingValue(name) => {
                let value = flag(name).and_then(|flag| flag.value).unwrap_or("VALUE");
                write!(f, "--{name} needs a value: --{name} <{value}>")?;
            }
            Self::UnexpectedValue(name) => write!(f, "--{name} takes no value")?,
            Self::Conflict(name) => write!(f, "--{name} and --no-{name} cannot both be given")?,
            Self::NotFor { flag, command } => {
                write!(f, "--{flag} is not an option of `{}`", command.name())?;
            }
            Self::BadValue { flag: name, value } => {
                write!(f, "--{name} cannot be {value:?}")?;
                if let Some(values) = flag(name).map(|flag| flag.values)
                    && !values.is_empty()
                {
                    write!(f, "; it is one of {}", values.join(", "))?;
                }
            }
        }
        write!(f, "\n\n{}", usage())
    }
}

impl core::error::Error for LaunchArgsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn a_bad_choice_lists_the_good_ones() {
        let text = LaunchArgsError::BadValue {
            flag: "graphics-quality",
            value: "max".into(),
        }
        .to_string();
        assert!(
            text.starts_with("--graphics-quality cannot be \"max\"; it is one of auto, low"),
            "{text}"
        );
        assert!(text.contains("Options:"), "{text}");
    }

    #[test]
    fn a_misplaced_flag_names_its_command() {
        let text = LaunchArgsError::NotFor {
            flag: "stop-at",
            command: LaunchCommand::Run,
        }
        .to_string();
        assert!(
            text.starts_with("--stop-at is not an option of `run`"),
            "{text}"
        );
        let text = LaunchArgsError::Conflict("vsync").to_string();
        assert!(text.starts_with("--vsync and --no-vsync"), "{text}");
    }

    #[test]
    fn help_is_the_usage_alone() {
        assert_eq!(LaunchArgsError::Help.to_string(), usage());
    }
}

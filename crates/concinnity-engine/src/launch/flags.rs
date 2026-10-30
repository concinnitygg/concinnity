// The commands and flags a launch accepts, described once for every front that
// offers them.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::gfx::view_modes::ShowFlags;

/// What a launch does with its world.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaunchCommand {
    /// Run the world.
    #[default]
    Run,
    /// Run the world's benchmark and print what its frames cost.
    Bench,
    /// Cook the world and stop.
    Cook,
}

impl LaunchCommand {
    /// Every command, in usage order.
    pub const ALL: [Self; 3] = [Self::Run, Self::Bench, Self::Cook];

    /// The name the command line gives the command.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Bench => "bench",
            Self::Cook => "cook",
        }
    }

    /// What the command does, in one line.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Run => "Run the app [default]",
            Self::Bench => "Benchmark the app and print what its frames cost when the run ends",
            Self::Cook => "Cook the app and exit",
        }
    }

    /// The command named `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|command| command.name() == name)
    }
}

/// One flag a launch accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchFlag {
    /// The long name, without its leading `--`.
    pub name: &'static str,
    /// The one-letter form, without its leading `-`.
    pub short: Option<char>,
    /// What the flag's value is called in usage text, or `None` for a switch
    /// that takes no value.
    pub value: Option<&'static str>,
    /// The values the flag accepts, or empty when any value of its kind does.
    pub values: &'static [&'static str],
    /// The value the flag takes when it is given without one, or `None` when
    /// a value is required.
    pub missing_value: Option<&'static str>,
    /// What the flag does, in one line.
    pub description: &'static str,
    /// Whether the flag may be given more than once, each value adding to the last.
    pub repeatable: bool,
    /// Whether `--no-<name>` turns the switch off. The two forms exclude each other.
    pub negatable: bool,
    /// The commands that take the flag.
    pub commands: &'static [LaunchCommand],
}

impl LaunchFlag {
    /// Whether `command` takes the flag.
    pub fn takes(&self, command: LaunchCommand) -> bool {
        self.commands.contains(&command)
    }

    // The flag's usage form, both spellings of a negatable switch included.
    fn usage_name(&self) -> String {
        let long = if self.negatable {
            format!("--[no-]{}", self.name)
        } else {
            format!("--{}", self.name)
        };
        let named = match self.short {
            Some(short) => format!("-{short}, {long}"),
            None => long,
        };
        match (self.value, self.missing_value) {
            (Some(value), Some(_)) => format!("{named} [<{value}>]"),
            (Some(value), None) => format!("{named} <{value}>"),
            (None, _) => named,
        }
    }
}

/// The port a debug server listens on when `--debug-port` names none.
pub const DEFAULT_DEBUG_PORT: u16 = 8777;

/// The quality presets `--graphics-quality` names, in [`QualityPreset::ALL`]
/// order.
///
/// [`QualityPreset::ALL`]: crate::settings::QualityPreset::ALL
pub const QUALITY_PRESET_VALUES: &[&str] = &["auto", "low", "medium", "high", "ultra", "custom"];

/// The graphics APIs `--graphics-api` names, by
/// [`Platform::key`](crate::platform::Platform::key).
pub const GRAPHICS_API_VALUES: &[&str] = &["metal", "directx", "vulkan"];

/// The window modes `--window-mode` names, by their authored names.
pub const WINDOW_MODE_VALUES: &[&str] =
    <crate::window::WindowMode as crate::components::Vocabulary>::VARIANTS;

/// The modes `--rt-dynamic` names, in
/// [`RtDynamicMode::ALL`](crate::render::rt_accel::geom::RtDynamicMode::ALL) order.
pub const RT_DYNAMIC_VALUES: &[&str] = &["off", "auto", "rebuild", "tlas"];

/// The presets `--dlss-preset` names, by their labels in lowercase.
pub const DLSS_PRESET_VALUES: &[&str] = &["default", "k", "l", "m"];

/// The passes `--hide-pass` names, by their [`ShowFlags::NAMED`] names.
pub const HIDE_PASS_VALUES: &[&str] = &{
    let mut names = [""; ShowFlags::NAMED.len()];
    let mut i = 0;
    while i < names.len() {
        names[i] = ShowFlags::NAMED[i].1;
        i += 1;
    }
    names
};

const ALL: &[LaunchCommand] = &LaunchCommand::ALL;
const RUNS: &[LaunchCommand] = &[LaunchCommand::Run, LaunchCommand::Bench];
const BENCH: &[LaunchCommand] = &[LaunchCommand::Bench];

const fn switch(name: &'static str, description: &'static str) -> LaunchFlag {
    LaunchFlag {
        name,
        short: None,
        value: None,
        values: &[],
        missing_value: None,
        description,
        repeatable: false,
        negatable: true,
        commands: RUNS,
    }
}

const fn valued(
    name: &'static str,
    value: &'static str,
    values: &'static [&'static str],
    description: &'static str,
) -> LaunchFlag {
    LaunchFlag {
        name,
        short: None,
        value: Some(value),
        values,
        missing_value: None,
        description,
        repeatable: false,
        negatable: false,
        commands: RUNS,
    }
}

/// Every flag a launch accepts.
pub const LAUNCH_FLAGS: &[LaunchFlag] = &[
    LaunchFlag {
        short: Some('q'),
        commands: ALL,
        negatable: false,
        ..switch("quiet", "Do not print log messages")
    },
    valued(
        "graphics-quality",
        "PRESET",
        QUALITY_PRESET_VALUES,
        "Force the graphics-quality preset for this run",
    ),
    valued(
        "graphics-api",
        "API",
        GRAPHICS_API_VALUES,
        "Render with this graphics API",
    ),
    valued(
        "window-mode",
        "MODE",
        WINDOW_MODE_VALUES,
        "Open the window in this mode",
    ),
    switch(
        "graphics-validation",
        "Turn the graphics API's validation layer on or off (default: on in debug builds)",
    ),
    switch("upscaling", "Turn temporal upscaling on or off"),
    valued(
        "rt-dynamic",
        "MODE",
        RT_DYNAMIC_VALUES,
        "How the ray-tracing acceleration structure tracks moving props",
    ),
    switch("rt-reflections", "Turn ray-traced reflections on or off"),
    switch(
        "rt-skinned-geometry",
        "Put skinned meshes in the ray-tracing acceleration structure or leave them out",
    ),
    switch("vsync", "Turn vertical sync on or off"),
    switch(
        "hud",
        "Start with the stats HUD shown or hidden (a toggle_hud key still flips it)",
    ),
    valued(
        "dlss-preset",
        "PRESET",
        DLSS_PRESET_VALUES,
        "Force the DLSS render preset",
    ),
    valued(
        "fixed-frame-rate",
        "RATE",
        &[],
        "Step the frame clock by exactly 1/RATE seconds per frame, whatever the wall time",
    ),
    valued(
        "screenshots-dir",
        "DIR",
        &[],
        "Write screenshots under DIR (default `screenshots`)",
    ),
    LaunchFlag {
        repeatable: true,
        ..valued(
            "hide-pass",
            "PASS",
            HIDE_PASS_VALUES,
            "Start with a feature pass off (repeatable)",
        )
    },
    valued("limit-frames", "N", &[], "Stop after N frames"),
    valued(
        "limit-fps",
        "N",
        &[],
        "Cap the frame rate at N frames a second (0 leaves it uncapped)",
    ),
    LaunchFlag {
        missing_value: Some("8777"),
        ..valued(
            "debug-port",
            "PORT",
            &[],
            "Serve the debug protocol on this localhost port (default 8777)",
        )
    },
    LaunchFlag {
        commands: BENCH,
        ..valued(
            "start-at",
            "SECONDS",
            &[],
            "Run the simulation ahead at the most ticks a frame allows until SECONDS into the run",
        )
    },
    LaunchFlag {
        commands: BENCH,
        ..valued("stop-at", "SECONDS", &[], "Stop the run SECONDS into it")
    },
    LaunchFlag {
        commands: BENCH,
        repeatable: true,
        ..valued(
            "screenshot-at",
            "SECONDS[:FILE]",
            &[],
            "Save the frame SECONDS into the run as FILE under the screenshot directory (repeatable)",
        )
    },
];

/// The flag named `name`, given without its leading `--`.
pub fn flag(name: &str) -> Option<&'static LaunchFlag> {
    LAUNCH_FLAGS.iter().find(|flag| flag.name == name)
}

/// The flag `-short` names.
pub fn short_flag(short: char) -> Option<&'static LaunchFlag> {
    LAUNCH_FLAGS.iter().find(|flag| flag.short == Some(short))
}

/// The usage text: the commands, then one line per flag with its name and
/// value, then what it does.
pub fn usage() -> String {
    let commands: Vec<(String, String)> = LaunchCommand::ALL
        .iter()
        .map(|command| {
            (
                command.name().to_string(),
                command.description().to_string(),
            )
        })
        .collect();
    let options: Vec<(String, String)> = LAUNCH_FLAGS
        .iter()
        .map(|flag| {
            let mut description = flag.description.to_string();
            if !flag.values.is_empty() {
                description = format!("{description} [{}]", flag.values.join(", "));
            }
            if flag.commands == BENCH {
                description.push_str(" (bench only)");
            }
            (flag.usage_name(), description)
        })
        .collect();
    let width = commands
        .iter()
        .chain(&options)
        .map(|(left, _)| left.len())
        .max()
        .unwrap_or(0);
    fmt::from_fn(|out| {
        out.write_str("Usage: [COMMAND] [OPTIONS]\n\nCommands:\n")?;
        for (left, description) in &commands {
            writeln!(out, "  {left:<width$}  {description}")?;
        }
        out.write_str("\nOptions:\n")?;
        writeln!(out, "  {:<width$}  Print help", "-h, --help")?;
        for (left, description) in &options {
            writeln!(out, "  {left:<width$}  {description}")?;
        }
        Ok(())
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::output::dlss::DlssPreset;
    use crate::render::rt_accel::geom::RtDynamicMode;
    use crate::settings::QualityPreset;

    #[test]
    fn flag_names_are_unique_and_bare() {
        for (i, a) in LAUNCH_FLAGS.iter().enumerate() {
            assert!(!a.name.starts_with('-'), "{}", a.name);
            assert!(!a.name.starts_with("no-"), "{}", a.name);
            assert!(LAUNCH_FLAGS[i + 1..].iter().all(|b| b.name != a.name));
            assert!(!a.commands.is_empty(), "{}", a.name);
        }
    }

    #[test]
    fn command_names_round_trip() {
        for command in LaunchCommand::ALL {
            assert_eq!(LaunchCommand::from_name(command.name()), Some(command));
        }
        assert_eq!(LaunchCommand::from_name("debug"), None);
    }

    // Only quiet reaches `cook`, which launches nothing.
    #[test]
    fn cook_takes_no_launch_flag_but_quiet() {
        let cook: Vec<&str> = LAUNCH_FLAGS
            .iter()
            .filter(|flag| flag.takes(LaunchCommand::Cook))
            .map(|flag| flag.name)
            .collect();
        assert_eq!(cook, ["quiet"]);
        assert_eq!(short_flag('q').map(|flag| flag.name), Some("quiet"));
    }

    #[test]
    fn the_value_lists_name_every_engine_value_in_order() {
        let presets: Vec<String> = QualityPreset::ALL
            .iter()
            .map(|p| p.name().to_ascii_lowercase())
            .collect();
        assert_eq!(presets, QUALITY_PRESET_VALUES);
        let dlss: Vec<String> = DlssPreset::ALL
            .iter()
            .map(|p| p.label().to_ascii_lowercase())
            .collect();
        assert_eq!(dlss, DLSS_PRESET_VALUES);
        let modes: Vec<&str> = RtDynamicMode::ALL.iter().map(|m| m.key()).collect();
        assert_eq!(modes, RT_DYNAMIC_VALUES);
        let passes: Vec<&str> = ShowFlags::NAMED.iter().map(|&(_, name)| name).collect();
        assert_eq!(passes, HIDE_PASS_VALUES);
    }

    #[test]
    fn the_debug_port_default_is_the_bare_flags_value() {
        let port = flag("debug-port").and_then(|flag| flag.missing_value);
        assert_eq!(port, Some(DEFAULT_DEBUG_PORT.to_string().as_str()));
    }

    #[test]
    fn usage_lists_every_command_and_flag() {
        let text = usage();
        for command in LaunchCommand::ALL {
            assert!(text.contains(command.name()), "{text}");
        }
        for flag in LAUNCH_FLAGS {
            assert!(text.contains(flag.name), "{text}");
            assert!(text.contains(flag.description), "{text}");
        }
        assert!(text.contains("--limit-fps <N>"), "{text}");
        assert!(text.contains("--[no-]vsync"), "{text}");
        assert!(text.contains("-q, --quiet"), "{text}");
        assert!(text.contains("--debug-port [<PORT>]"), "{text}");
        assert!(text.contains("(bench only)"), "{text}");
        assert!(
            text.contains("[auto, low, medium, high, ultra, custom]"),
            "{text}"
        );
    }
}

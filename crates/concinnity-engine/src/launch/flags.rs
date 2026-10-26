// The flags a launch accepts, described once for every front that offers them.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// One flag a launch accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchFlag {
    /// The long name, without its leading `--`.
    pub name: &'static str,
    /// What the flag's value is called in usage text, or `None` for a switch
    /// that takes no value.
    pub value: Option<&'static str>,
    /// The values the flag accepts, or empty when any value of its kind does.
    pub values: &'static [&'static str],
    /// What the flag does, in one line.
    pub description: &'static str,
    /// Whether the flag may be given more than once, each value adding to the last.
    pub repeatable: bool,
}

/// The quality presets `--quality-preset` names, in [`QualityPreset::ALL`]
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

/// Every flag a launch accepts.
pub const LAUNCH_FLAGS: &[LaunchFlag] = &[
    LaunchFlag {
        name: "benchmark",
        value: None,
        values: &[],
        description: "Benchmark the world and print what its frames cost when the run ends",
        repeatable: false,
    },
    LaunchFlag {
        name: "vsync",
        value: None,
        values: &[],
        description: "Turn vertical sync on, whatever the world or the saved settings say",
        repeatable: false,
    },
    LaunchFlag {
        name: "fps-cap",
        value: Some("N"),
        values: &[],
        description: "Cap the frame rate at N frames a second (0 leaves it uncapped)",
        repeatable: false,
    },
    LaunchFlag {
        name: "quality-preset",
        value: Some("PRESET"),
        values: QUALITY_PRESET_VALUES,
        description: "Force the graphics-quality preset for this run, unsaved",
        repeatable: false,
    },
    LaunchFlag {
        name: "graphics-api",
        value: Some("API"),
        values: GRAPHICS_API_VALUES,
        description: "Render with this graphics API, over the saved settings' choice, unsaved",
        repeatable: false,
    },
    LaunchFlag {
        name: "window-mode",
        value: Some("MODE"),
        values: WINDOW_MODE_VALUES,
        description: "Open the window in this mode, over the world's and the saved settings' choice, unsaved",
        repeatable: false,
    },
    LaunchFlag {
        name: "screenshots-dir",
        value: Some("DIR"),
        values: &[],
        description: "Write screenshots under DIR (default `screenshots`)",
        repeatable: false,
    },
    LaunchFlag {
        name: "screenshots",
        value: None,
        values: &[],
        description: "Take the world's authored screenshots, as a benchmark always does",
        repeatable: false,
    },
    LaunchFlag {
        name: "screenshot-at",
        value: Some("SECONDS[:FILE]"),
        values: &[],
        description: "Save the frame SECONDS into the run as FILE under the screenshot directory (repeatable)",
        repeatable: true,
    },
    LaunchFlag {
        name: "frames",
        value: Some("N"),
        values: &[],
        description: "Stop after N frames",
        repeatable: false,
    },
];

/// The flag named `name`, given without its leading `--`.
pub fn flag(name: &str) -> Option<&'static LaunchFlag> {
    LAUNCH_FLAGS.iter().find(|flag| flag.name == name)
}

/// One line per flag: its name and value, then what it does.
pub fn usage() -> String {
    let rows: Vec<(String, String)> = LAUNCH_FLAGS
        .iter()
        .map(|flag| {
            let left = match flag.value {
                Some(value) => format!("--{} <{value}>", flag.name),
                None => format!("--{}", flag.name),
            };
            let description = if flag.values.is_empty() {
                flag.description.to_string()
            } else {
                format!("{} [{}]", flag.description, flag.values.join(", "))
            };
            (left, description)
        })
        .collect();
    let width = rows.iter().map(|(left, _)| left.len()).max().unwrap_or(0);
    fmt::from_fn(|out| {
        out.write_str("Options:\n")?;
        for (left, description) in &rows {
            writeln!(out, "  {left:<width$}  {description}")?;
        }
        Ok(())
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::QualityPreset;

    #[test]
    fn flag_names_are_unique_and_bare() {
        for (i, a) in LAUNCH_FLAGS.iter().enumerate() {
            assert!(!a.name.starts_with('-'), "{}", a.name);
            assert!(LAUNCH_FLAGS[i + 1..].iter().all(|b| b.name != a.name));
        }
    }

    #[test]
    fn the_preset_values_name_every_preset_in_order() {
        let names: Vec<String> = QualityPreset::ALL
            .iter()
            .map(|p| p.name().to_ascii_lowercase())
            .collect();
        assert_eq!(names, QUALITY_PRESET_VALUES);
    }

    #[test]
    fn usage_lists_every_flag_with_its_value_and_choices() {
        let text = usage();
        for flag in LAUNCH_FLAGS {
            assert!(text.contains(&format!("--{}", flag.name)), "{text}");
            assert!(text.contains(flag.description), "{text}");
        }
        assert!(text.contains("--fps-cap <N>"), "{text}");
        assert!(
            text.contains("[auto, low, medium, high, ultra, custom]"),
            "{text}"
        );
    }
}

// The flags a launch accepts, described once for every front that offers them.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

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
}

/// The quality presets `--quality-preset` names, in [`QualityPreset::ALL`]
/// order.
///
/// [`QualityPreset::ALL`]: crate::settings::QualityPreset::ALL
pub const QUALITY_PRESET_VALUES: &[&str] = &["auto", "low", "medium", "high", "ultra", "custom"];

/// Every flag a launch accepts.
pub const LAUNCH_FLAGS: &[LaunchFlag] = &[
    LaunchFlag {
        name: "benchmark",
        value: None,
        values: &[],
        description: "Benchmark the world and print what its frames cost when the run ends",
    },
    LaunchFlag {
        name: "vsync",
        value: None,
        values: &[],
        description: "Turn vertical sync on, whatever the world or the saved settings say",
    },
    LaunchFlag {
        name: "fps-cap",
        value: Some("N"),
        values: &[],
        description: "Cap the frame rate at N frames a second (0 leaves it uncapped)",
    },
    LaunchFlag {
        name: "quality-preset",
        value: Some("PRESET"),
        values: QUALITY_PRESET_VALUES,
        description: "Force the graphics-quality preset for this run, unsaved",
    },
    LaunchFlag {
        name: "frames",
        value: Some("N"),
        values: &[],
        description: "Stop after N frames",
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
            let mut left = format!("--{}", flag.name);
            if let Some(value) = flag.value {
                let _ = write!(left, " <{value}>");
            }
            let mut description = String::from(flag.description);
            if !flag.values.is_empty() {
                let _ = write!(description, " [{}]", flag.values.join(", "));
            }
            (left, description)
        })
        .collect();
    let width = rows.iter().map(|(left, _)| left.len()).max().unwrap_or(0);
    let mut out = String::from("Options:\n");
    for (left, description) in rows {
        let _ = writeln!(out, "  {left:<width$}  {description}");
    }
    out
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

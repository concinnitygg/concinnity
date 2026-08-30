// `cn version`, and the line `cn --version` prints. Both render through
// `version_line` so the subcommand and the flag can never drift apart.
//
// The commit and the date come from the build stamp the build script bakes in.

use std::sync::OnceLock;

/// The engine version this build was compiled from.
///
/// Every crate in the workspace shares it, so it names the toolchain as a
/// whole rather than any one library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// The commit the source was built from, empty when the tree was not a
// checkout, and the date: the commit's when there is one, the build's when
// there is not.
const COMMIT: &str = env!("CONCINNITY_COMMIT");
const STAMP_DATE: &str = env!("CONCINNITY_STAMP_DATE");

// The name the CLI is invoked under, and what clap prefixes its own
// `--version` output with.
const NAME: &str = "concinnity";

/// Print the version.
pub fn version() -> std::io::Result<()> {
    println!("{}", version_line());
    Ok(())
}

/// The version and its build stamp, as `--version` reports them after the
/// command name: `0.19.0 (c980f4866 2026-06-30)`.
///
/// Borrowed for the process because that is what clap's version field takes.
pub fn version_details() -> &'static str {
    static DETAILS: OnceLock<String> = OnceLock::new();
    DETAILS.get_or_init(|| details(VERSION, COMMIT, STAMP_DATE))
}

/// The full one-line version banner, command name included.
pub fn version_line() -> String {
    format!("{NAME} {}", version_details())
}

// A build off a checkout names its commit, so the date beside it is that
// commit's. Otherwise the only date there is the day of the build, and it is
// labelled rather than left to read as a commit date.
fn details(version: &str, commit: &str, date: &str) -> String {
    match (commit.is_empty(), date.is_empty()) {
        (false, false) => format!("{version} ({commit} {date})"),
        (false, true) => format!("{version} ({commit})"),
        (true, false) => format!("{version} ({date})"),
        (true, true) => version.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_commit_build_names_the_commit_and_its_date() {
        assert_eq!(
            details("0.19.0", "c980f4866", "2026-06-30"),
            "0.19.0 (c980f4866 2026-06-30)"
        );
    }

    #[test]
    fn a_build_off_no_checkout_labels_the_date_as_the_build_date() {
        assert_eq!(details("0.19.0", "", "2026-08-30"), "0.19.0 (2026-08-30)");
    }

    #[test]
    fn a_stamp_missing_a_part_drops_only_that_part() {
        assert_eq!(details("0.19.0", "c980f4866", ""), "0.19.0 (c980f4866)");
        assert_eq!(details("0.19.0", "", ""), "0.19.0");
    }

    // clap renders `--version` as "{name} {details}", so the banner has to be
    // that same pair; nothing else keeps the two spellings identical.
    #[test]
    fn the_line_is_the_name_then_the_details() {
        assert_eq!(version_line(), format!("concinnity {}", version_details()));
        assert!(version_details().starts_with(VERSION));
    }

    #[test]
    fn the_version_is_a_dotted_release() {
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert_eq!(parts.len(), 3, "{VERSION} is not major.minor.patch");
        assert!(
            parts[0].chars().all(|c| c.is_ascii_digit()),
            "{VERSION} has a non-numeric major"
        );
    }

    // The stamp is baked by the build script; an empty date means it stopped
    // running, which would silently strip the build info from every release.
    #[test]
    fn the_build_stamp_carries_a_date() {
        assert!(!STAMP_DATE.is_empty(), "the build script stamped no date");
    }
}

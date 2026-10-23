// The authored screenshot: when in the run, what file, and whether the HUD
// shows in it.

use alloc::string::String;

/// Saves the frame drawn at a moment of the run as a PNG image, in a run
/// that takes screenshots.
///
/// The moment is counted on the fixed simulation clock from when the world
/// starts, a benchmark's warm-up included. A camera track, a benchmark's
/// route and anything else driven by that clock is in the same pose at the
/// same moment on every machine, so a screenshot shows the same thing however
/// fast the machine draws. The frame saved is the first one drawn at or after
/// the moment. A moment the run never reaches is reported when the run ends.
///
/// `file` is a relative path under the run's screenshot directory. It cannot
/// leave that directory or name a directory: an absolute path, one with a
/// `..` part or a `:`, or one ending in `/` is refused, and so are two
/// screenshots naming the same file.
///
/// ```json
/// ["Screenshot", { "seconds": 25.0, "file": "beach.png", "hud": false }]
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Screenshot {
    /// Seconds from the start of the run, on the simulation clock.
    pub seconds: f32,
    /// The image's path under the screenshot directory.
    pub file: String,
    /// Whether the HUD and other UI show in the image.
    #[asset(default = true)]
    pub hud: bool,
}

/// Whether `file` names a file inside the screenshot directory: not empty,
/// not absolute, with no `..` part and no `:` anywhere, and ending in a file
/// name rather than a directory.
pub fn is_contained_file(file: &str) -> bool {
    let mut parts = file.split(['/', '\\']);
    let named = parts
        .next_back()
        .is_some_and(|last| !matches!(last, "" | "." | ".."));
    named
        && !file.starts_with(['/', '\\'])
        && !file.contains(':')
        && !file.split(['/', '\\']).any(|part| part == "..")
}

/// `file` as one spelling of the path it names, so two spellings of one file
/// compare equal: `/` between parts, no empty or `.` parts, and lowercase,
/// since the file systems screenshots land on mostly ignore case.
pub fn normal_file(file: &str) -> String {
    let mut normal = String::with_capacity(file.len());
    for part in file
        .split(['/', '\\'])
        .filter(|part| !matches!(*part, "" | "."))
    {
        if !normal.is_empty() {
            normal.push('/');
        }
        normal.extend(part.chars().map(|c| c.to_ascii_lowercase()));
    }
    normal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_stays_inside_the_screenshot_directory() {
        assert!(is_contained_file("beach.png"));
        assert!(is_contained_file("benchmark/beach.png"));
        assert!(is_contained_file("a..b.png"));
        assert!(is_contained_file("./a/b.png"));
        for escaping in [
            "",
            ".",
            "shots/",
            "shots/.",
            "/tmp/a.png",
            "\\\\host\\a.png",
            "C:/a.png",
            "a.png:stream",
            "../a.png",
            "a/../../b.png",
            "a\\..\\b.png",
        ] {
            assert!(!is_contained_file(escaping), "{escaping}");
        }
    }

    #[test]
    fn spellings_of_one_file_normalize_alike() {
        assert_eq!(normal_file("./Shots\\a.png"), "shots/a.png");
        assert_eq!(normal_file("shots//./A.PNG"), "shots/a.png");
        assert_ne!(normal_file("shots/a.png"), normal_file("a.png"));
    }

    #[test]
    fn a_bare_screenshot_shows_the_hud() {
        let shot: Screenshot =
            crate::test_support::from_json(r#"{"seconds": 3.0, "file": "a.png"}"#);
        assert!(shot.hud);
        assert_eq!(shot.seconds, 3.0);
    }
}

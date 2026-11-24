// World text together with the file it was read from and the root its paths
// resolve against.

use std::path::Path;

/// World file text, the file it was read from, and the root its relative
/// paths resolve against.
///
/// The root is the build's: every file path in the world is rebased onto it,
/// and it is the fallback for a file with no library root of its own. A world
/// with neither can only name absolute paths.
#[derive(Debug, Clone, Copy)]
pub struct WorldSource<'a> {
    /// The file's text.
    pub text: &'a str,
    /// The file the text was read from.
    pub file: Option<&'a Path>,
    /// The build's root.
    pub root: Option<&'a Path>,
}

impl<'a> WorldSource<'a> {
    /// Text read from `file`, built under `root`.
    pub fn file(text: &'a str, file: &'a Path, root: Option<&'a Path>) -> Self {
        Self {
            text,
            file: Some(file),
            root,
        }
    }

    /// Text held in memory, built under `root`.
    pub fn text(text: &'a str, root: Option<&'a Path>) -> Self {
        Self {
            text,
            file: None,
            root,
        }
    }
}

impl<'a> From<&'a str> for WorldSource<'a> {
    fn from(text: &'a str) -> Self {
        Self {
            text,
            file: None,
            root: None,
        }
    }
}

impl<'a> From<&'a String> for WorldSource<'a> {
    fn from(text: &'a String) -> Self {
        text.as_str().into()
    }
}

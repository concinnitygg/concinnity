// World text together with the file it was read from, which is what an
// `Include` line in it resolves against.

use std::path::Path;

/// World file text, and the file it was read from when there is one.
///
/// A world built from a string alone has no file, so an `Include` in it can
/// only name an absolute path; [`WorldSource::file`] lets a relative one
/// resolve beside the world file, as it does in a build from disk.
#[derive(Debug, Clone, Copy)]
pub struct WorldSource<'a> {
    /// The world file's text.
    pub text: &'a str,
    /// The file the text was read from.
    pub file: Option<&'a Path>,
}

impl<'a> WorldSource<'a> {
    /// Text read from `file`.
    pub fn file(text: &'a str, file: &'a Path) -> Self {
        Self {
            text,
            file: Some(file),
        }
    }
}

impl<'a> From<&'a str> for WorldSource<'a> {
    fn from(text: &'a str) -> Self {
        Self { text, file: None }
    }
}

impl<'a> From<&'a String> for WorldSource<'a> {
    fn from(text: &'a String) -> Self {
        text.as_str().into()
    }
}

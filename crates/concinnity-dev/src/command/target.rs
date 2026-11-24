//! Which file a subcommand operates on.

use std::path::{Path, PathBuf};

use crate::command::app::{apps_under, no_application, select};

/// What a command was pointed at: a file named with `-f`, or the project's
/// application, picked by `--app` when there are several.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Target<'a> {
    pub(crate) file: Option<&'a str>,
    pub(crate) app: Option<&'a str>,
}

impl<'a> Target<'a> {
    /// The file at `file`.
    pub(crate) fn file(file: &'a str) -> Self {
        Self {
            file: Some(file),
            app: None,
        }
    }

    /// The file this target names: the `-f` path, else the application's file.
    ///
    /// # Errors
    ///
    /// Returns an error when the `-f` file does not exist, or when there is no
    /// application or no single one to pick.
    pub(crate) fn resolve(self) -> std::io::Result<PathBuf> {
        self.find()?.ok_or_else(no_application)
    }

    /// The file `-f` or `--app` names, else `None`, without falling back to the
    /// only application.
    ///
    /// # Errors
    ///
    /// Returns an error when the `-f` file does not exist, or `--app` names no
    /// single application.
    pub(crate) fn named(self) -> std::io::Result<Option<PathBuf>> {
        if self.file.is_none() && self.app.is_none() {
            return Ok(None);
        }
        self.resolve().map(Some)
    }

    /// The `-f` path, else the application's file, else `None` when the
    /// project has no application and none was asked for.
    ///
    /// A `-f` that does not exist is an error rather than a fall back to
    /// discovery, which would run a different file under the name the caller
    /// asked for, and say nothing.
    ///
    /// # Errors
    ///
    /// Returns an error when the `-f` file does not exist, or there is no single
    /// application to pick.
    pub(crate) fn find(self) -> std::io::Result<Option<PathBuf>> {
        if let Some(file) = self.file {
            return if Path::new(file).exists() {
                Ok(Some(PathBuf::from(file)))
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("file not found: {file}"),
                ))
            };
        }
        let root = crate::project::root_or_cwd();
        Ok(select(apps_under(&root), self.app, &root)?.map(|app| app.path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_existing_file_is_taken_as_given() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("scene.jsonl");
        std::fs::write(&file, "").expect("write");
        let given = file.to_string_lossy().into_owned();
        assert_eq!(Target::file(&given).resolve().expect("resolves"), file);
    }

    // A named file that is not there fails loudly instead of silently becoming
    // whatever discovery turns up.
    #[test]
    fn a_missing_file_errors_rather_than_falling_back() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("absent.jsonl");
        let given = missing.to_string_lossy().into_owned();
        let err = Target::file(&given)
            .resolve()
            .expect_err("missing file is an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("absent.jsonl"), "{err}");
    }

    // Without `-f`, the project's application is found under its root.
    #[test]
    fn without_a_file_the_projects_application_is_found() {
        let _guard = crate::test_support::lock();
        let dir = concinnity_testing::TempTree::new();
        crate::project::open(concinnity_host::store::paths::StateTree::at(dir.path()));
        let none = Target::default().find();
        let err = Target::default().resolve().unwrap_err();

        let app = dir.path().join("apps").join("arena.jsonl");
        std::fs::create_dir_all(app.parent().unwrap()).unwrap();
        std::fs::write(&app, "[\"AppConfig\",{\"name\":\"Arena\"}]\n").unwrap();
        let found = Target::default().resolve();
        let named = Target {
            app: Some("Arena"),
            ..Target::default()
        }
        .resolve();
        crate::test_support::isolate_state_dir();

        assert_eq!(none.unwrap(), None);
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(found.unwrap(), app);
        assert_eq!(named.unwrap(), app);
    }
}

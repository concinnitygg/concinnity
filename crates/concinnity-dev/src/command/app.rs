//! The applications under a project: each `.jsonl` file that declares an
//! `AppConfig` of its own, and which one a command means.

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::declared_app_name;

use crate::command::data_files::{data_files, project_outputs};

/// One application: the name its `AppConfig` declares, and the file declaring it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct App {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
}

/// Every application under `root`, in path order. A file that cannot be read
/// is not one; `check` reports it.
pub(crate) fn apps_under(root: &Path) -> Vec<App> {
    let app_config = RegisteredType::AppConfig.as_str();
    data_files(root, &project_outputs())
        .files
        .into_iter()
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).ok()?;
            // Most files are not applications, and naming the type is cheaper
            // than parsing every line to find that out.
            if !text.contains(app_config) {
                return None;
            }
            let name = declared_app_name(&text)?;
            Some(App { name, path })
        })
        .collect()
}

/// The application among `apps` a command means: the one named `wanted`, or
/// the only one when no name is given. `None` when there is none and none was
/// asked for. Paths in errors are shown relative to `root`.
///
/// # Errors
///
/// Returns an error listing the applications when several match, or when
/// `wanted` names none of them.
pub(crate) fn select(
    apps: Vec<App>,
    wanted: Option<&str>,
    root: &Path,
) -> std::io::Result<Option<App>> {
    let listing = |apps: &[App]| {
        use std::fmt::Write;
        let width = apps.iter().map(|a| a.name.len()).max().unwrap_or(0);
        let mut out = String::new();
        for a in apps {
            let path = a.path.strip_prefix(root).unwrap_or(&a.path);
            #[expect(
                clippy::let_underscore_must_use,
                reason = "writing to a String cannot fail"
            )]
            let _ = write!(out, "\n  {:<width$}  {}", a.name, path.display());
        }
        out
    };
    let Some(wanted) = wanted else {
        return match apps.len() {
            0 | 1 => Ok(apps.into_iter().next()),
            n => Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "found {n} applications; pick one with `--app <name>`:{}",
                    listing(&apps)
                ),
            )),
        };
    };
    let (mut named, others): (Vec<App>, Vec<App>) =
        apps.into_iter().partition(|a| a.name == wanted);
    match named.len() {
        1 => Ok(named.pop()),
        0 if others.is_empty() => Err(Error::new(
            ErrorKind::NotFound,
            format!("no application named '{wanted}': no file declares an AppConfig"),
        )),
        0 => Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "no application named '{wanted}'; found:{}",
                listing(&others)
            ),
        )),
        n => Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{n} applications are named '{wanted}'; name one with `-f <file>`:{}",
                listing(&named)
            ),
        )),
    }
}

/// The error a command that needs an application reports when the project
/// has none.
pub(crate) fn no_application() -> Error {
    Error::new(
        ErrorKind::NotFound,
        "no application found: declare an AppConfig in a .jsonl file, run \
         `concinnity init`, or name a file with `-f`",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(name: &str, path: &str) -> App {
        App {
            name: name.to_string(),
            path: PathBuf::from(path),
        }
    }

    fn write(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn only_files_declaring_an_app_config_are_applications() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "apps/arena.jsonl",
            "[\"AppConfig\",{\"name\":\"Arena\"}]\n",
        );
        write(
            dir.path(),
            "apps/garden.jsonl",
            "[\"AppConfig\",{\"name\":\"Garden\"}]\n",
        );
        write(dir.path(), "props/kit.jsonl", "[\"Logger\",{}]\n");
        write(
            dir.path(),
            "wrapper.jsonl",
            "[\"Include\",{\"path\":\"apps/arena.jsonl\"}]\n",
        );
        assert_eq!(
            apps_under(dir.path()),
            [
                app(
                    "Arena",
                    &dir.path().join("apps/arena.jsonl").to_string_lossy()
                ),
                app(
                    "Garden",
                    &dir.path().join("apps/garden.jsonl").to_string_lossy()
                ),
            ]
        );
    }

    #[test]
    fn the_only_application_is_picked_without_a_name() {
        let picked = select(vec![app("Arena", "a.jsonl")], None, Path::new("")).unwrap();
        assert_eq!(picked, Some(app("Arena", "a.jsonl")));
    }

    #[test]
    fn no_application_and_no_name_is_none() {
        assert_eq!(select(Vec::new(), None, Path::new("")).unwrap(), None);
    }

    #[test]
    fn several_applications_need_a_name_and_are_listed() {
        let apps = vec![
            app("Arena", "/p/apps/a.jsonl"),
            app("Garden", "/p/apps/g.jsonl"),
        ];
        let err = select(apps.clone(), None, Path::new("/p")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        let message = err.to_string();
        assert!(message.contains("--app"), "{message}");
        assert!(message.contains("Arena   apps/a.jsonl"), "{message}");
        assert!(message.contains("Garden  apps/g.jsonl"), "{message}");

        let picked = select(apps, Some("Garden"), Path::new("/p")).unwrap();
        assert_eq!(picked, Some(app("Garden", "/p/apps/g.jsonl")));
    }

    #[test]
    fn an_unknown_name_lists_the_applications_there_are() {
        let err = select(vec![app("Arena", "a.jsonl")], Some("Nope"), Path::new("")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(err.to_string().contains("Arena"), "{err}");

        let err = select(Vec::new(), Some("Nope"), Path::new("")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn a_name_two_applications_share_asks_for_a_file() {
        let apps = vec![app("Arena", "a.jsonl"), app("Arena", "b.jsonl")];
        let err = select(apps, Some("Arena"), Path::new("")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.to_string().contains("-f"), "{err}");
    }
}

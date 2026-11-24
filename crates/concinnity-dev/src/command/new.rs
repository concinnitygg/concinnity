use concinnity_cook::authoring::world::APP_JSONL;
use std::path::Path;

use crate::command::app::apps_under;

// The starter application's assets after its AppConfig. Everything else a
// running application needs (window, renderer, debug HUD) is injected at build
// time and recorded in cook-lock.json; `cn list --expanded` shows the effective
// set.
//
// The label names no Font, so it draws with the engine's built-in face. It asks
// for `centered` itself rather than leaning on a default: unset, the greeting
// lands at the label's default x/y, under the HUD chips in the top-left corner.
const STARTER_ASSETS: &str = r#"["TextLabel",{"$id":"hello_world","content":"Hello, world!","centered":true}]
"#;

/// Create a new project in a new directory at `path`.
pub(crate) fn new(path: &Path) -> std::io::Result<()> {
    // A pre-existing empty directory is a valid target; one that already holds
    // an application is not.
    if let Some(app) = apps_under(path).first() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "'{}' already contains the application '{}' in {}",
                path.display(),
                app.name,
                app.path.display()
            ),
        ));
    }
    std::fs::create_dir_all(path)?;
    println!("Created directory '{}'", path.display());
    init_in_dir(path)
}

/// Create a new project in the working directory.
pub(crate) fn init() -> std::io::Result<()> {
    init_in_dir(Path::new("."))
}

/// The `AppConfig` line that starts an application in `dir`, named after it.
pub(super) fn app_config_line(dir: &Path) -> String {
    let name = std::fs::canonicalize(dir)
        .ok()
        .and_then(|dir| dir.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    serde_json::json!(["AppConfig", {"name": name}]).to_string()
}

// Write the starter application into `dir` and run an initial build.
fn init_in_dir(dir: &Path) -> std::io::Result<()> {
    if let Some(app) = apps_under(dir).first() {
        println!(
            "{} already declares the application '{}', skipping init",
            app.path.display(),
            app.name
        );
        return Ok(());
    }
    let app_path = dir.join(APP_JSONL);
    if app_path.exists() {
        println!("{} already exists, skipping init", app_path.display());
        return Ok(());
    }

    let text = format!("{}\n{STARTER_ASSETS}", app_config_line(dir));
    std::fs::write(&app_path, text)?;
    println!("Created {}", app_path.display());

    crate::authoring::build_world_file(&app_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the paths that stop before the initial build are exercised here;
    // a successful `cn new` runs the full compile pipeline.

    #[test]
    fn new_refuses_a_directory_that_already_has_an_application() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("arena.jsonl"),
            "[\"AppConfig\",{\"name\":\"Arena\"}]\n",
        )
        .unwrap();

        let err = new(dir.path()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains("Arena"), "got: {err}");
    }

    #[test]
    fn init_in_dir_skips_when_an_application_exists() {
        let dir = tempfile::tempdir().unwrap();
        let app = dir.path().join("arena.jsonl");
        let text = "[\"AppConfig\",{\"name\":\"Arena\"}]\n";
        std::fs::write(&app, text).unwrap();

        init_in_dir(dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&app).unwrap(), text);
        assert!(!dir.path().join(APP_JSONL).exists());
    }

    #[test]
    fn init_in_dir_leaves_an_existing_app_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let app = dir.path().join(APP_JSONL);
        std::fs::write(&app, "[\"Logger\",{\"$id\":\"keep\"}]\n").unwrap();

        init_in_dir(dir.path()).unwrap();
        let content = std::fs::read_to_string(&app).unwrap();
        assert!(content.contains("\"keep\""), "got: {content}");
    }

    #[test]
    fn the_starter_app_config_is_named_after_its_directory() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("my-app");
        std::fs::create_dir(&dir).unwrap();
        assert_eq!(app_config_line(&dir), r#"["AppConfig",{"name":"my-app"}]"#);
    }
}

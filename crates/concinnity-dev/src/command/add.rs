//! Target wrapper around `crate::authoring::add_to_path`.
//!
//! Adds to the file `-f` names, else the project's application. With no
//! application, it adds to `app.jsonl` in the working directory and declares
//! the `AppConfig` that makes it one when the file lacks it.

use concinnity_cook::authoring::world::{
    APP_JSONL, declared_app_name, parse_entry, patch_world_jsonl_to,
};

use crate::authoring::add_to_path;
use crate::command::Target;
use crate::command::new::app_config_line;
use std::path::{Path, PathBuf};

/// Add the assets `targets` name to the file `target` names.
pub(crate) fn add(
    target: Target<'_>,
    name: Option<&str>,
    targets: &[String],
    template: Option<&str>,
) -> std::io::Result<()> {
    if let Some(path) = target.find()? {
        return add_to_path(&path, name, targets, template);
    }
    let path = PathBuf::from(APP_JSONL);
    add_to_path(&path, name, targets, template)?;
    if declared_app_name(&std::fs::read_to_string(&path)?).is_none() {
        declare_app(&path)?;
    }
    Ok(())
}

// Make the file at `path` an application, named after its directory.
fn declare_app(path: &Path) -> std::io::Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let entry = parse_entry(&app_config_line(dir))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    patch_world_jsonl_to(path, path, |assets| {
        assets.insert(0, entry);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_application_file_declares_its_app_config_first() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("garden");
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join(APP_JSONL);
        std::fs::write(&path, "[\"Logger\",{\"$id\":\"log\"}]\n").unwrap();

        declare_app(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let first = text.lines().next().unwrap();
        assert_eq!(first, r#"["AppConfig",{"name":"garden"}]"#);
        assert!(text.contains("\"log\""), "{text}");
    }
}

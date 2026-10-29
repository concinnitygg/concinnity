// `Screenshot` checks: each names a file inside the screenshot directory, and
// no two name the same one.

use std::collections::HashMap;

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use concinnity_engine::screenshot::{is_contained_file, normal_file};

/// A screenshot's `file` must be a relative path that stays inside the
/// screenshot directory.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
    if is_contained_file(file) {
        Ok(())
    } else {
        Err(format!(
            "Screenshot '{name}': `file` {file:?} must be a relative path to a file inside the screenshot directory, with no `..` part or `:`"
        ))
    }
}

/// Two screenshots writing one file would overwrite each other.
pub(crate) fn check_unique_files(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let mut seen: HashMap<String, &str> = HashMap::new();
    for asset in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Screenshot)
    {
        let Some(file) = asset.args.get("file").and_then(|v| v.as_str()) else {
            continue;
        };
        if let Some(first) = seen.insert(normal_file(file), asset.id.as_str()) {
            errors.push(format!(
                "Screenshots '{first}' and '{}' both write {file:?}; give each its own file",
                asset.id
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(name: &str, file: &str) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: name.to_string(),
            asset_type: RegisteredType::Screenshot,
            args: serde_json::json!({ "seconds": 1.0, "file": file }),
        }
    }

    #[test]
    fn a_file_must_stay_in_the_screenshot_directory() {
        assert!(check("a", &serde_json::json!({"file": "beach.png"})).is_ok());
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"file": "../beach.png"}),
            serde_json::json!({"file": "/tmp/beach.png"}),
            serde_json::json!({"file": "beach.png:stream"}),
            serde_json::json!({"file": "shots/"}),
        ] {
            assert!(
                check("a", &bad)
                    .unwrap_err()
                    .contains("inside the screenshot directory")
            );
        }
    }

    #[test]
    fn two_screenshots_cannot_share_a_file() {
        let mut errors = Vec::new();
        check_unique_files(&[shot("a", "x.png"), shot("b", "y.png")], &mut errors);
        assert!(errors.is_empty());
        check_unique_files(
            &[shot("a", "x.png"), shot("b", "./sub/../x.png")],
            &mut errors,
        );
        assert!(errors.is_empty(), "refused by the file check instead");
        check_unique_files(&[shot("a", "x.png"), shot("b", "./x.png")], &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("'a' and 'b'"), "{}", errors[0]);
    }
}

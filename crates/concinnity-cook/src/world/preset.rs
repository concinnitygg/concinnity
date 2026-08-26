// Preset file loading utilities shared by the expansion modules. A preset is a
// named JSON snippet under the build's asset search root, in a `<subdir>/`
// (palettes, prefabs, light_rigs, shots) that an authored asset references by
// name; the cook pipeline inlines it at build time. Build-only, so it lives
// here rather than in the runtime foundation.

use std::path::Path;

fn find_preset_path(filename: &str, subdir: &str, assets_dir: Option<&Path>) -> Option<String> {
    let assets = assets_dir?;
    if let Some(p) = crate::source::find_in(assets, filename) {
        return Some(p);
    }
    preset_path_in(assets, filename, subdir)
}

// The `<assets>/<subdir>/<filename>` path when it exists. Split out so the
// direct-path rule is testable apart from the recursive search.
fn preset_path_in(assets: &Path, filename: &str, subdir: &str) -> Option<String> {
    let path = assets.join(subdir).join(filename);
    if path.exists() {
        return Some(path.to_string_lossy().into_owned());
    }
    None
}

/// Load a JSON object from `<assets_dir>/<subdir>/<name>.json`.
pub fn load_preset_obj(name: &str, subdir: &str, assets_dir: Option<&Path>) -> serde_json::Value {
    let filename = format!("{}.json", name);
    let path = find_preset_path(&filename, subdir, assets_dir);
    let Some(path) = path else {
        return serde_json::Value::Null;
    };
    read_preset_json(&path)
}

// A preset file's contents, or Null when it cannot be read or parsed: a
// malformed preset falls back to the type defaults rather than failing the
// build. Split out so both outcomes are testable against a temp file.
fn read_preset_json(path: &str) -> serde_json::Value {
    let Ok(content) = std::fs::read_to_string(path) else {
        return serde_json::Value::Null;
    };
    serde_json::from_str::<serde_json::Value>(&content).unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_preset_obj_reads_the_named_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let rigs = dir.path().join("light_rigs");
        std::fs::create_dir_all(&rigs).unwrap();
        std::fs::write(rigs.join("noon.json"), r#"{"args":{"lights":[]}}"#).unwrap();

        let v = load_preset_obj("noon", "light_rigs", Some(dir.path()));
        assert!(v["args"]["lights"].is_array());
        // A name with no preset file, and a build with no asset search root at
        // all, both read as Null.
        assert!(load_preset_obj("absent", "light_rigs", Some(dir.path())).is_null());
        assert!(load_preset_obj("noon", "light_rigs", None).is_null());
    }

    #[test]
    fn preset_path_resolves_under_the_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let shots = dir.path().join("shots");
        std::fs::create_dir_all(&shots).unwrap();
        std::fs::write(shots.join("wide.json"), "{}").unwrap();

        let found = preset_path_in(dir.path(), "wide.json", "shots").expect("preset path");
        assert_eq!(found, shots.join("wide.json").to_string_lossy());
        // A file in another subdir is not this preset.
        assert_eq!(preset_path_in(dir.path(), "wide.json", "palettes"), None);
        assert_eq!(preset_path_in(dir.path(), "other.json", "shots"), None);
    }

    #[test]
    fn preset_json_parses_into_a_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rig.json");
        std::fs::write(&path, r#"{"args":{"lights":[{"kind":"point"}]}}"#).unwrap();
        let v = read_preset_json(path.to_str().unwrap());
        assert_eq!(v["args"]["lights"][0]["kind"], "point");
    }

    // A malformed preset reads as Null, so the referencing asset falls back to
    // its type defaults instead of failing the build.
    #[test]
    fn malformed_preset_json_reads_as_null() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(read_preset_json(path.to_str().unwrap()).is_null());
    }

    #[test]
    fn unreadable_preset_path_reads_as_null() {
        assert!(read_preset_json("/no/such/preset.json").is_null());
    }
}

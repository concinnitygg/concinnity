// Preset file loading utilities shared by the expansion modules. A preset is a
// named JSON snippet under `.concinnity/assets/<subdir>/` (palettes, prefabs,
// light_rigs, shots) that an authored asset references by name; the cook
// pipeline inlines it at build time. Build-only, so it lives here rather than
// in the runtime foundation.

fn find_preset_path(filename: &str, subdir: &str) -> Option<String> {
    if let Some(p) = crate::paths::find_in_assets(filename) {
        return Some(p);
    }
    let path = crate::paths::assets_dir().join(subdir).join(filename);
    if path.exists() {
        return Some(path.to_string_lossy().into_owned());
    }
    None
}

// Load a JSON object from assets/<subdir>/<name>.json.
pub fn load_preset_obj(name: &str, subdir: &str) -> serde_json::Value {
    let filename = format!("{}.json", name);
    let path = find_preset_path(&filename, subdir);
    let Some(path) = path else {
        return serde_json::Value::Null;
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return serde_json::Value::Null;
    };
    serde_json::from_str::<serde_json::Value>(&content).unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the miss path is exercised here: the hit path resolves against the
    // process-global assets-dir anchor, which paths.rs tests may redirect
    // concurrently, so pointing it at a temp tree in this test would race them.
    #[test]
    fn load_preset_obj_returns_null_when_absent() {
        let v = load_preset_obj("cn_test_no_such_preset", "cn_test_subdir");
        assert!(v.is_null());
    }
}

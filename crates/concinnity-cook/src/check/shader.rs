/// Check a `Shader`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate Shader args without compiling: the `fragment` file is required and
// both declared files must name a shader source.
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    check_file(args, "fragment", true)?;
    check_file(args, "vertex", false)
}

fn check_file(args: &serde_json::Value, field: &str, required: bool) -> Result<(), String> {
    let Some(value) = args.get(field).filter(|v| !v.is_null()) else {
        if required {
            return Err(format!(
                "Shader requires a `{field}` file: a `.hlsl` path defining `shade`"
            ));
        }
        return Ok(());
    };
    let path = value
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("Shader `{field}` must be a non-empty `.hlsl` path"))?;
    if is_shader_source(path) {
        Ok(())
    } else {
        Err(format!(
            "Shader `{field}` is '{path}', which is not a `.hlsl` file; a Shader is written \
             in HLSL, one source for every backend"
        ))
    }
}

fn is_shader_source(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_shader_extension)
}

/// True for the extension of a shader source file, in any case, so a `.HLSL`
/// path is accepted everywhere a `.hlsl` one is.
pub fn is_shader_extension(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("hlsl")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_fragment_file_is_required_and_a_vertex_file_is_not() {
        assert!(check_args(&json!({"fragment": "a.hlsl"})).is_ok());
        assert!(check_args(&json!({"vertex": "v.hlsl", "fragment": "a.hlsl"})).is_ok());
        let err = check_args(&json!({"vertex": "v.hlsl"})).unwrap_err();
        assert!(err.contains("`fragment`"), "got: {err}");
        let err = check_args(&json!({})).unwrap_err();
        assert!(err.contains("`fragment`"), "got: {err}");
    }

    // The per-platform table and per-backend languages are gone: a declaration
    // still spelling either is refused with a message that says why.
    #[test]
    fn a_file_in_no_shader_language_or_the_old_table_is_refused() {
        let err = check_args(&json!({"fragment": "a.metal"})).unwrap_err();
        assert!(err.contains("not a `.hlsl` file"), "got: {err}");
        let err = check_args(&json!({"fragment": {"sources": {"metal": "a.metal"}}})).unwrap_err();
        assert!(err.contains("non-empty `.hlsl` path"), "got: {err}");
        let err = check_args(&json!({"fragment": "a.hlsl", "vertex": ""})).unwrap_err();
        assert!(err.contains("`vertex`"), "got: {err}");
    }

    // HLSL is the one shader language, so any other extension is refused.
    #[test]
    fn only_an_hlsl_file_is_accepted() {
        let err = check_args(&json!({"vertex": "v.glsl", "fragment": "a.hlsl"})).unwrap_err();
        assert!(err.contains("not a `.hlsl` file"), "got: {err}");
        let err = check_args(&json!({"fragment": "a.glsl"})).unwrap_err();
        assert!(err.contains("not a `.hlsl` file"), "got: {err}");
    }

    // `cn add` and the hot-reload watcher both take the extension in any case,
    // so the check does too: a file they accept is never refused here.
    #[test]
    fn the_extension_is_matched_in_any_case() {
        assert!(check_args(&json!({"fragment": "a.HLSL", "vertex": "v.Hlsl"})).is_ok());
    }

    // The named form prefixes the asset name so a world-wide report says which
    // asset failed.
    #[test]
    fn the_error_names_the_asset() {
        let err = check("scene_shader", &json!({})).unwrap_err();
        assert!(err.starts_with("Asset 'scene_shader':"), "got: {err}");
    }
}

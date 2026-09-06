/// Check a `Shader`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}

// Validate Shader args without compiling: the `fragment` file is required and
// both declared files must be `.slang` paths.
fn check_args(args: &serde_json::Value) -> Result<(), String> {
    check_file(args, "fragment", true)?;
    check_file(args, "vertex", false)
}

fn check_file(args: &serde_json::Value, field: &str, required: bool) -> Result<(), String> {
    let Some(value) = args.get(field).filter(|v| !v.is_null()) else {
        if required {
            return Err(format!(
                "Shader requires a `{field}` file: a `.slang` path defining `shade`"
            ));
        }
        return Ok(());
    };
    let path = value
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("Shader `{field}` must be a non-empty `.slang` path"))?;
    if std::path::Path::new(path)
        .extension()
        .is_some_and(|e| e == "slang")
    {
        Ok(())
    } else {
        Err(format!(
            "Shader `{field}` is '{path}', which is not a `.slang` file; a Shader is written \
             in Slang, one source for every backend"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_fragment_file_is_required_and_a_vertex_file_is_not() {
        assert!(check_args(&json!({"fragment": "a.slang"})).is_ok());
        assert!(check_args(&json!({"vertex": "v.slang", "fragment": "a.slang"})).is_ok());
        let err = check_args(&json!({"vertex": "v.slang"})).unwrap_err();
        assert!(err.contains("`fragment`"), "got: {err}");
        let err = check_args(&json!({})).unwrap_err();
        assert!(err.contains("`fragment`"), "got: {err}");
    }

    // The per-platform table and per-backend languages are gone: a declaration
    // still spelling either is refused with a message that says why.
    #[test]
    fn a_non_slang_file_or_the_old_table_is_refused() {
        let err = check_args(&json!({"fragment": "a.metal"})).unwrap_err();
        assert!(err.contains("not a `.slang` file"), "got: {err}");
        let err = check_args(&json!({"fragment": {"sources": {"metal": "a.metal"}}})).unwrap_err();
        assert!(err.contains("non-empty `.slang` path"), "got: {err}");
        let err = check_args(&json!({"fragment": "a.slang", "vertex": ""})).unwrap_err();
        assert!(err.contains("`vertex`"), "got: {err}");
    }

    // The named form prefixes the asset name so a world-wide report says which
    // asset failed.
    #[test]
    fn the_error_names_the_asset() {
        let err = check("scene_shader", &json!({})).unwrap_err();
        assert!(err.starts_with("Asset 'scene_shader':"), "got: {err}");
    }
}

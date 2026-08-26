pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    // A `.glb`-sourced Mesh has no inline vertices yet: the build's desugar
    // pass fills them in later. Skip the compile-time check; full validation
    // happens when desugar reads the GLB.
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if !source.is_empty() {
        return Ok(());
    }
    // Full compile catches unknown generators and structural errors. Pure
    // geometry math, no I/O -- the one source-reading generator (heightfield)
    // needs a `source`, which the early return above already excluded -- so
    // the check needs no asset search root.
    crate::mesh_compile::compile_mesh_payload(args, None)
        .map(|_| ())
        .map_err(|e| format!("Asset '{}' mesh compile error: {}", name, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glb_sourced_mesh_skips_the_compile_check() {
        // A file-backed Mesh has no inline geometry yet; the check is skipped.
        let args = serde_json::json!({ "source": "model.glb", "primitive_index": 0 });
        assert!(check("m", &args).is_ok());
    }

    #[test]
    fn valid_generator_compiles_clean() {
        let args = serde_json::json!({ "generator": "sphere", "radius": 1.0 });
        assert!(check("m", &args).is_ok());
    }

    #[test]
    fn unknown_generator_maps_to_a_compile_error() {
        let args = serde_json::json!({ "generator": "definitely_not_real" });
        let err = check("m", &args).unwrap_err();
        assert!(err.contains("mesh compile error"), "got: {err}");
    }
}

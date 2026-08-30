//! Structural validation for VoxelChunk args. Cross-asset palette lookups are
//! handled by crate::check::cross_reference::validate_cross_references; this check
//! only catches problems we can see from the chunk's own args alone.

/// Check a `VoxelChunk`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let dim = args.get("dim").and_then(|v| v.as_array()).ok_or_else(|| {
        format!(
            "Asset '{}': VoxelChunk `dim` must be a [dx, dy, dz] array",
            name
        )
    })?;
    if dim.len() < 3 {
        return Err(format!(
            "Asset '{}': VoxelChunk `dim` must have 3 elements, got {}",
            name,
            dim.len()
        ));
    }
    let dims: [u64; 3] = [
        dim[0].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[0] must be a non-negative integer",
                name
            )
        })?,
        dim[1].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[1] must be a non-negative integer",
                name
            )
        })?,
        dim[2].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[2] must be a non-negative integer",
                name
            )
        })?,
    ];
    let expected = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);

    let blocks = args
        .get("blocks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk `blocks` must be an array of palette indices",
                name
            )
        })?;
    if (blocks.len() as u64) != expected {
        return Err(format!(
            "Asset '{}': VoxelChunk blocks length {} does not match dim {}x{}x{} ({} expected)",
            name,
            blocks.len(),
            dims[0],
            dims[1],
            dims[2],
            expected
        ));
    }

    let palette_len = args
        .get("palette")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    for (i, entry) in blocks.iter().enumerate() {
        let idx = entry.as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk blocks[{}] must be a non-negative integer",
                name, i
            )
        })?;
        if palette_len == 0 || (idx as usize) >= palette_len {
            return Err(format!(
                "Asset '{}': VoxelChunk blocks[{}] = {} out of palette range (len {})",
                name, i, idx, palette_len
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ok_args() -> serde_json::Value {
        json!({
            "dim": [2, 1, 2],
            "blocks": [0, 1, 0, 1],
            "palette": ["air", "stone"]
        })
    }

    #[test]
    fn valid_chunk_passes() {
        check("c", &ok_args()).expect("valid chunk");
    }

    #[test]
    fn zero_dim_with_empty_blocks_passes() {
        let args = json!({"dim": [0, 4, 4], "blocks": [], "palette": ["air"]});
        check("c", &args).expect("empty chunk");
    }

    #[test]
    fn missing_dim_errors() {
        let err = check("c", &json!({"blocks": []})).unwrap_err();
        assert!(
            err.contains("`dim` must be a [dx, dy, dz] array"),
            "got: {err}"
        );
    }

    #[test]
    fn short_dim_errors() {
        let mut args = ok_args();
        args["dim"] = json!([2, 2]);
        let err = check("c", &args).unwrap_err();
        assert!(err.contains("must have 3 elements, got 2"), "got: {err}");
    }

    #[test]
    fn non_integer_dim_element_errors() {
        let mut args = ok_args();
        args["dim"] = json!([2, "one", 2]);
        let err = check("c", &args).unwrap_err();
        assert!(
            err.contains("dim[1] must be a non-negative integer"),
            "got: {err}"
        );
    }

    #[test]
    fn negative_dim_element_errors() {
        let mut args = ok_args();
        args["dim"] = json!([2, 1, -2]);
        let err = check("c", &args).unwrap_err();
        assert!(
            err.contains("dim[2] must be a non-negative integer"),
            "got: {err}"
        );
    }

    #[test]
    fn missing_blocks_errors() {
        let args = json!({"dim": [1, 1, 1], "palette": ["air"]});
        let err = check("c", &args).unwrap_err();
        assert!(err.contains("`blocks` must be an array"), "got: {err}");
    }

    #[test]
    fn blocks_length_mismatch_errors() {
        let mut args = ok_args();
        args["blocks"] = json!([0, 1, 0]);
        let err = check("c", &args).unwrap_err();
        assert!(
            err.contains("blocks length 3 does not match dim 2x1x2 (4 expected)"),
            "got: {err}"
        );
    }

    #[test]
    fn non_integer_block_errors() {
        let mut args = ok_args();
        args["blocks"] = json!([0, 1, "x", 1]);
        let err = check("c", &args).unwrap_err();
        assert!(
            err.contains("blocks[2] must be a non-negative integer"),
            "got: {err}"
        );
    }

    #[test]
    fn block_index_past_the_palette_errors() {
        let mut args = ok_args();
        args["blocks"] = json!([0, 1, 2, 1]);
        let err = check("c", &args).unwrap_err();
        assert!(
            err.contains("blocks[2] = 2 out of palette range (len 2)"),
            "got: {err}"
        );
    }

    #[test]
    fn any_block_with_an_empty_palette_errors() {
        let args = json!({"dim": [1, 1, 1], "blocks": [0]});
        let err = check("c", &args).unwrap_err();
        assert!(err.contains("out of palette range (len 0)"), "got: {err}");
    }

    #[test]
    fn error_messages_carry_the_asset_name() {
        let err = check("cave_chunk", &json!({})).unwrap_err();
        assert!(err.contains("Asset 'cave_chunk'"), "got: {err}");
    }
}

// Readers for the optional numeric args a generator takes from world.jsonl.

use concinnity_core::math::narrow_f32;

// `key` as an f32, or `default` when it is absent or not a number.
pub(super) fn f32_arg(args: &serde_json::Value, key: &str, default: f32) -> f32 {
    args.get(key)
        .and_then(serde_json::Value::as_f64)
        .map_or(default, narrow_f32)
}

// `key` as a u32, or `default` when it is absent or not an unsigned integer. A value past
// `u32::MAX` is an authoring error rather than something to clamp.
pub(super) fn u32_arg(args: &serde_json::Value, key: &str, default: u32) -> Result<u32, String> {
    match args.get(key).and_then(serde_json::Value::as_u64) {
        None => Ok(default),
        Some(v) => u32::try_from(v).map_err(|_| format!("{key} = {v} is out of range")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn f32_arg_reads_or_defaults() {
        let args = json!({"radius": 2.5, "name": "x"});
        assert_eq!(f32_arg(&args, "radius", 1.0), 2.5);
        assert_eq!(f32_arg(&args, "height", 1.0), 1.0);
        assert_eq!(f32_arg(&args, "name", 1.0), 1.0);
    }

    #[test]
    fn u32_arg_rejects_values_past_u32() {
        let args = json!({"segments": 24, "rings": 5_000_000_000_u64, "neg": -1});
        assert_eq!(u32_arg(&args, "segments", 16), Ok(24));
        assert_eq!(u32_arg(&args, "missing", 16), Ok(16));
        assert_eq!(u32_arg(&args, "neg", 16), Ok(16));
        assert!(u32_arg(&args, "rings", 16).unwrap_err().contains("rings"));
    }
}

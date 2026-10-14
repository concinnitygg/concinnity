// Structural validation for WaterSurface args: what a planet sea can use.

/// Check a `WaterSurface`'s authored args.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let planet_sea = args.get("shape").and_then(|v| v.as_str()) == Some("planet_sea");
    let swell = args
        .get("waves")
        .and_then(|v| v.as_array())
        .is_some_and(|w| !w.is_empty());
    if planet_sea && swell {
        return Err(format!(
            "Asset '{name}': WaterSurface is a `planet_sea` with authored `waves`, which a planet \
             sea does not support; its waves all come from the wind, so drop the swell"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_planet_sea_with_swell_is_refused() {
        let args = serde_json::json!({"shape": "planet_sea", "waves": [{"amplitude": 0.2}]});
        let err = check("sea", &args).unwrap_err();
        assert!(err.contains("planet_sea") && err.contains("waves"), "{err}");
    }

    #[test]
    fn swell_on_a_rectangle_and_a_bare_planet_sea_pass() {
        let rect = serde_json::json!({"waves": [{"amplitude": 0.2}]});
        assert!(check("pond", &rect).is_ok());
        let sea = serde_json::json!({"shape": "planet_sea", "sea_level": 12.0, "waves": []});
        assert!(check("sea", &sea).is_ok());
    }
}

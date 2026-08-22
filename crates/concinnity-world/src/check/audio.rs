//! Audio authoring checks: emitter attenuation ranges and the shared volume /
//! bus / rolloff fields on emitters and cues.

const BUS_NAMES: [&str; 3] = ["music", "sfx", "voice"];
const ROLLOFF_NAMES: [&str; 3] = ["logarithmic", "linear", "none"];

fn check_volume(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let Some(volume) = args.get("volume").and_then(|v| v.as_f64()) else {
        return Ok(());
    };
    if volume.is_finite() && volume >= 0.0 {
        return Ok(());
    }
    Err(format!(
        "Asset '{name}': volume must be a non-negative gain, got {volume}"
    ))
}

fn check_bus(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let Some(bus) = args.get("bus").and_then(|v| v.as_str()) else {
        return Ok(());
    };
    if BUS_NAMES.contains(&bus) {
        return Ok(());
    }
    Err(format!(
        "Asset '{}': unknown bus '{}'; expected one of {}",
        name,
        bus,
        BUS_NAMES.join(", ")
    ))
}

// AudioEmitter: min/max distance must form a positive, finite, non-empty
// range, and the rolloff must be a known curve.
pub(crate) fn check_emitter(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_volume(name, args)?;
    check_bus(name, args)?;
    if let Some(rolloff) = args.get("rolloff").and_then(|v| v.as_str())
        && !ROLLOFF_NAMES.contains(&rolloff)
    {
        return Err(format!(
            "Asset '{}': unknown rolloff '{}'; expected one of {}",
            name,
            rolloff,
            ROLLOFF_NAMES.join(", ")
        ));
    }
    let field = |key: &str| args.get(key).and_then(|v| v.as_f64());
    let min = field("min_distance");
    let max = field("max_distance");
    for (key, value) in [("min_distance", min), ("max_distance", max)] {
        if let Some(value) = value
            && !(value.is_finite() && value >= 0.0)
        {
            return Err(format!(
                "Asset '{name}': {key} must be a non-negative distance, got {value}"
            ));
        }
    }
    // Defaults fill whichever bound is unauthored, mirroring the schema.
    let min = min.unwrap_or(1.0);
    let max = max.unwrap_or(50.0);
    if max <= min {
        return Err(format!(
            "Asset '{name}': max_distance ({max}) must exceed min_distance ({min})"
        ));
    }
    Ok(())
}

// AudioCue: volume and bus follow the shared rules.
pub(crate) fn check_cue(name: &str, args: &serde_json::Value) -> Result<(), String> {
    check_volume(name, args)?;
    check_bus(name, args)
}

// PropBody: the impact gain must be a non-negative finite number.
pub(crate) fn check_prop_body(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let Some(volume) = args.get("impact_volume").and_then(|v| v.as_f64()) else {
        return Ok(());
    };
    if volume.is_finite() && volume >= 0.0 {
        return Ok(());
    }
    Err(format!(
        "Asset '{name}': impact_volume must be a non-negative gain, got {volume}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_emitters_pass() {
        assert!(check_emitter("e", &json!({})).is_ok(), "defaults are valid");
        let full = json!({
            "min_distance": 2.0, "max_distance": 80.0,
            "rolloff": "linear", "bus": "voice", "volume": 0.5
        });
        assert!(check_emitter("e", &full).is_ok());
    }

    #[test]
    fn degenerate_distance_ranges_are_rejected() {
        for args in [
            json!({"min_distance": 10.0, "max_distance": 10.0}),
            json!({"min_distance": 10.0, "max_distance": 3.0}),
            // Authored min above the default max of 50.
            json!({"min_distance": 60.0}),
            // Authored max below the default min of 1.
            json!({"max_distance": 0.5}),
            json!({"min_distance": -1.0}),
        ] {
            assert!(check_emitter("e", &args).is_err(), "rejects {args}");
        }
    }

    #[test]
    fn unknown_rolloff_and_bus_names_are_rejected() {
        assert!(check_emitter("e", &json!({"rolloff": "inverse"})).is_err());
        assert!(check_emitter("e", &json!({"bus": "ambience"})).is_err());
        assert!(check_cue("c", &json!({"bus": "ambience"})).is_err());
        assert!(check_cue("c", &json!({"bus": "voice"})).is_ok());
    }

    #[test]
    fn negative_volumes_are_rejected() {
        assert!(check_emitter("e", &json!({"volume": -0.5})).is_err());
        assert!(check_cue("c", &json!({"volume": -1.0})).is_err());
        assert!(check_cue("c", &json!({"volume": 0.0})).is_ok());
    }

    #[test]
    fn impact_volume_follows_the_gain_rules() {
        assert!(check_prop_body("b", &json!({})).is_ok());
        assert!(check_prop_body("b", &json!({"impact_volume": 0.5})).is_ok());
        assert!(check_prop_body("b", &json!({"impact_volume": -1.0})).is_err());
    }
}

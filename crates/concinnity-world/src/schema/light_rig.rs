//! Named light-grouping schema.

/// A named grouping of lights.
///
/// Use `preset` to expand a built-in setup into named
/// [DirectionalLight](#directionallight)/[PointLight](#pointlight) assets
/// (`<rig_name>_<light_name>`), or declare lights directly and list their names
/// in `lights`.
///
/// **Library presets:**
///
/// ```rust
/// # use concinnity_world::registry::build_only::LightRig;
/// LightRig {
///     preset: "rig_outdoor_sun_fill".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LightRig {
    /// Name of a built-in or file-backed preset (e.g. "rig_outdoor_sun_fill").
    /// When set, `lights` is ignored.
    pub preset: String,
    /// Names of existing [DirectionalLight](#directionallight) or
    /// [PointLight](#pointlight) assets to include in this rig. Ignored when
    /// `preset` is set.
    pub lights: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_rig_groups_nothing() {
        let r = LightRig::default();
        assert!(r.preset.is_empty());
        assert!(r.lights.is_empty());
    }

    #[test]
    fn an_inline_rig_lists_the_lights_it_groups() {
        let r: LightRig = serde_json::from_str(r#"{"lights":["sun","fill"]}"#).unwrap();
        assert_eq!(r.lights, ["sun", "fill"]);
        assert!(r.preset.is_empty());
    }

    #[test]
    fn a_preset_rig_leaves_the_light_list_empty() {
        let r: LightRig = serde_json::from_str(r#"{"preset":"rig_outdoor_sun_fill"}"#).unwrap();
        assert_eq!(r.preset, "rig_outdoor_sun_fill");
        assert!(r.lights.is_empty());

        let bytes = postcard::to_allocvec(&r).unwrap();
        let back: LightRig = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.preset, "rig_outdoor_sun_fill");
        assert!(back.lights.is_empty());
    }
}

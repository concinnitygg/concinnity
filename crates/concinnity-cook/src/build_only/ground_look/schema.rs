//! Sand and dirt surfaces drawn by the cook.

use concinnity_engine::components::{DirtLook, SandLook};

/// A sand surface, drawn by the cook into seamless one-meter textures.
///
/// Expands into a [Material](#material) of the same name, so a
/// [Planet](#planet)'s `material` or a cover's `material` names it directly,
/// and the two [Texture](#texture)s that material samples:
/// `<name>_albedo` (base color, with the relief's height in alpha) and
/// `<name>_normal`. The textures repeat every UV unit; a planet's ground maps
/// one unit to a meter.
///
/// ```rust
/// # use concinnity_cook::authoring::registry::build_only::Sand;
/// # use concinnity_engine::components::SandLook;
/// Sand {
///     look: SandLook {
///         ripple_direction_deg: 30.0,
///         shells: 0.2,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    concinnity_engine::ecs::AssetFields,
    concinnity_engine::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Sand {
    /// What the sand looks like.
    #[serde(flatten)]
    pub look: SandLook,
    /// Texels across each one-meter tile of its textures.
    #[asset(default = 1024)]
    pub resolution: u32,
}

/// A dirt surface, drawn by the cook into seamless one-meter textures.
///
/// Expands into a [Material](#material) of the same name, so a
/// [Planet](#planet)'s `material` or a cover's `material` names it directly,
/// and the two [Texture](#texture)s that material samples:
/// `<name>_albedo` (base color, with the relief's height in alpha) and
/// `<name>_normal`. The textures repeat every UV unit; a planet's ground maps
/// one unit to a meter.
///
/// ```rust
/// # use concinnity_cook::authoring::registry::build_only::Dirt;
/// # use concinnity_engine::components::DirtLook;
/// Dirt {
///     look: DirtLook {
///         stones: 0.4,
///         cracks: 0.3,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    concinnity_engine::ecs::AssetFields,
    concinnity_engine::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Dirt {
    /// What the dirt looks like.
    #[serde(flatten)]
    pub look: DirtLook,
    /// Texels across each one-meter tile of its textures.
    #[asset(default = 1024)]
    pub resolution: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sands_look_sits_at_its_top_level() {
        let sand: Sand = serde_json::from_str(r#"{"shells":0.3,"resolution":256}"#).unwrap();
        assert_eq!(sand.look.shells, 0.3);
        assert_eq!(sand.resolution, 256);
        assert_eq!(sand.look.ripple_spacing, SandLook::default().ripple_spacing);
    }

    #[test]
    fn a_blank_dirt_takes_the_default_look() {
        let dirt: Dirt = serde_json::from_str("{}").unwrap();
        assert_eq!(dirt.look, DirtLook::default());
        assert_eq!(dirt.resolution, 1024);
    }
}

// Material-palette schema: a named set of Material entries with short aliases.

use alloc::string::String;
use alloc::vec::Vec;

/// A named set of [Material](#material) entries with short aliases.
///
/// Expands into [Material](#material) assets named `<palette_name>_<alias>`.
/// [Prop](#prop)s reference the expanded names.
///
/// ```jsonl
/// // Inline:
/// {"type":"MaterialPalette","name":"pal","args":{"entries":[
///   {"alias":"floor","albedo":"tex_stone","roughness":0.9},
///   {"alias":"wall","albedo":"tex_brick","roughness":0.85},
///   {"alias":"trim","albedo":"tex_metal","roughness":0.3,"metallic":0.8}
/// ]}}
/// // Props reference expanded names:
/// {"type":"Prop","name":"floor","args":{"mesh":"room_mesh","material":"pal_floor","position":[0,0,0]}}
///
/// // From library preset:
/// {"type":"MaterialPalette","name":"pal","args":{"preset":"pal_stone_dungeon"}}
/// {"type":"Prop","name":"south_wall","args":{"mesh":"wall_mesh","material":"pal_wall"}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MaterialPalette {
    /// Name of a built-in or file-backed preset (e.g. "pal_stone_dungeon").
    /// When set, `entries` is ignored.
    pub preset: String,
    /// Inline material entries. Ignored when `preset` is set.
    pub entries: Vec<PaletteEntry>,
}

/// One entry in a [MaterialPalette]. Each carries an `alias` (the suffix of the
/// expanded [Material](#material) name) plus the Material fields the expansion
/// fills in. Names in `albedo` / `normal_map` are unresolved [Texture](#texture)
/// references, resolved on the expanded Material.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PaletteEntry {
    /// Alias suffix; the expanded material is named `<palette>_<alias>`.
    pub alias: String,
    /// [Texture](#texture) name for the material's albedo.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub albedo: String,
    /// [Texture](#texture) name for the material's normal map.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub normal_map: String,
    /// Surface roughness in [0, 1].
    pub roughness: f32,
    /// Metallic factor in [0, 1].
    pub metallic: f32,
    /// Linear-space RGB tint multiplier.
    pub tint: [f32; 3],
    /// Linear-space RGB emissive factor.
    pub emissive_factor: [f32; 3],
}

impl Default for PaletteEntry {
    fn default() -> Self {
        Self {
            alias: String::from("surface"),
            albedo: String::new(),
            normal_map: String::new(),
            roughness: 0.8,
            metallic: 0.0,
            tint: [1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_deserialises_with_material_defaults() {
        let e: PaletteEntry =
            serde_json::from_str(r#"{"alias":"floor","albedo":"tex_stone","roughness":0.9}"#)
                .unwrap();
        assert_eq!(e.alias, "floor");
        assert_eq!(e.albedo, "tex_stone");
        assert_eq!(e.roughness, 0.9);
        // Omitted fields fall back to the material defaults the expansion uses.
        assert_eq!(e.metallic, 0.0);
        assert_eq!(e.tint, [1.0, 1.0, 1.0]);
        assert_eq!(e.emissive_factor, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn preset_palette_leaves_entries_empty() {
        let p: MaterialPalette = serde_json::from_str(r#"{"preset":"pal_stone_dungeon"}"#).unwrap();
        assert_eq!(p.preset, "pal_stone_dungeon");
        assert!(p.entries.is_empty());
    }
}

// Graphics settings schema.

use crate::components::{
    AmbientOcclusion, AmbientOcclusionOverrides, AntiAliasing, AntiAliasingOverrides, Background,
    BackgroundOverrides, Bloom, BloomOverrides, Culling, CullingOverrides, Display,
    DisplayOverrides, Exposure, ExposureOverrides, FramePacing, FramePacingOverrides,
    IndirectLighting, IndirectLightingOverrides, PlanetDetail, PlanetDetailOverrides, Reflections,
    ReflectionsOverrides, Shadows, ShadowsOverrides, TextureFiltering, TextureFilteringOverrides,
    Vignette, VignetteOverrides,
};

/// The world's graphics settings, one section per feature. One per world; a
/// world that draws gets an empty one when it declares none.
///
/// Every field of every section is optional. A field left unset follows the
/// graphics quality preset: under `Auto` the detected GPU picks a tier, and the
/// tier caps the quality-governed fields (shadow resolution and reach, texture
/// filtering, ambient occlusion, reflections, the screen-space bounce,
/// anti-aliasing, the upscaler's render scale and planet terrain detail) below
/// the section defaults.
/// A field set here replaces the default but stays under the same cap on
/// weaker hardware, and a choice the player makes in the settings menu
/// replaces both.
///
/// ```json
/// ["GraphicsConfig", {
///   "shadows": { "map_size": 2048, "distance": 120 },
///   "anti_aliasing": { "mode": "fxaa" },
///   "background": { "color": [0.1, 0.1, 0.15, 1.0] }
/// }]
/// ```
///
/// ```rust
/// # use concinnity_engine::components::{GraphicsConfig, ShadowsOverrides};
/// GraphicsConfig {
///     shadows: ShadowsOverrides {
///         map_size: Some(2048),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default, deny_unknown_fields)]
pub struct GraphicsConfig {
    /// Frames in flight, display sync and the frame-rate cap.
    #[asset(section)]
    pub frame_pacing: FramePacingOverrides,
    /// Cascaded shadow maps for directional lights.
    #[asset(section)]
    pub shadows: ShadowsOverrides,
    /// Surface texture filtering.
    #[asset(section)]
    pub texture_filtering: TextureFilteringOverrides,
    /// The color shown where nothing draws.
    #[asset(section)]
    pub background: BackgroundOverrides,
    /// Screen-space ambient occlusion.
    #[asset(section)]
    pub ambient_occlusion: AmbientOcclusionOverrides,
    /// Screen-space and ray-traced reflections.
    #[asset(section)]
    pub reflections: ReflectionsOverrides,
    /// The ambient term and the screen-space bounce.
    #[asset(section)]
    pub indirect_lighting: IndirectLightingOverrides,
    /// Anti-aliasing and temporal upscaling.
    #[asset(section)]
    pub anti_aliasing: AntiAliasingOverrides,
    /// Fixed and automatic exposure.
    #[asset(section)]
    pub exposure: ExposureOverrides,
    /// Bloom.
    #[asset(section)]
    pub bloom: BloomOverrides,
    /// Corner darkening.
    #[asset(section)]
    pub vignette: VignetteOverrides,
    /// HDR and PQ output.
    #[asset(section)]
    pub display: DisplayOverrides,
    /// Visibility culling.
    #[asset(section)]
    pub culling: CullingOverrides,
    /// Planet terrain tile detail.
    #[asset(section)]
    pub planet_detail: PlanetDetailOverrides,
}

impl GraphicsConfig {
    /// Every field of every section set to that section's default: what an
    /// empty `GraphicsConfig` resolves to before the quality tier caps it.
    pub fn explicit_defaults() -> Self {
        Self {
            frame_pacing: FramePacing::default().into(),
            shadows: Shadows::default().into(),
            texture_filtering: TextureFiltering::default().into(),
            background: Background::default().into(),
            ambient_occlusion: AmbientOcclusion::default().into(),
            reflections: Reflections::default().into(),
            indirect_lighting: IndirectLighting::default().into(),
            anti_aliasing: AntiAliasing::default().into(),
            exposure: Exposure::default().into(),
            bloom: Bloom::default().into(),
            vignette: Vignette::default().into(),
            display: Display::default().into(),
            culling: Culling::default().into(),
            planet_detail: PlanetDetail::default().into(),
        }
    }
}

// A readable format (the authored JSON) leaves out sections with nothing set;
// a binary one (the cooked blob) writes every section in order.
impl serde::Serialize for GraphicsConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let readable = serializer.is_human_readable();
        let mut state = serializer.serialize_struct("GraphicsConfig", GRAPHICS_SECTIONS.len())?;
        macro_rules! sections {
            ($($field:ident),*) => {$(
                if readable && self.$field.is_unset() {
                    state.skip_field(stringify!($field))?;
                } else {
                    state.serialize_field(stringify!($field), &self.$field)?;
                }
            )*};
        }
        sections!(
            frame_pacing,
            shadows,
            texture_filtering,
            background,
            ambient_occlusion,
            reflections,
            indirect_lighting,
            anti_aliasing,
            exposure,
            bloom,
            vignette,
            display,
            culling,
            planet_detail
        );
        state.end()
    }
}

/// The type name of every section `GraphicsConfig` holds, with its key.
pub const GRAPHICS_SECTIONS: [(&str, &str); 14] = [
    ("FramePacing", "frame_pacing"),
    ("Shadows", "shadows"),
    ("TextureFiltering", "texture_filtering"),
    ("Background", "background"),
    ("AmbientOcclusion", "ambient_occlusion"),
    ("Reflections", "reflections"),
    ("IndirectLighting", "indirect_lighting"),
    ("AntiAliasing", "anti_aliasing"),
    ("Exposure", "exposure"),
    ("Bloom", "bloom"),
    ("Vignette", "vignette"),
    ("Display", "display"),
    ("Culling", "culling"),
    ("PlanetDetail", "planet_detail"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{AaMode, ShadowUpdate, Shadows};
    use crate::ecs::asset_fields::{AssetFields, ClampedParam};
    use alloc::vec::Vec;

    #[test]
    fn an_absent_key_is_unset_and_a_default_value_is_set() {
        let parsed: GraphicsConfig =
            serde_json::from_str(r#"{"shadows":{"map_size":4096},"bloom":{}}"#).unwrap();
        assert_eq!(parsed.shadows.map_size, Some(4096));
        assert_eq!(parsed.shadows.distance, None);
        assert!(parsed.bloom.is_unset());
        assert!(parsed.anti_aliasing.is_unset());
    }

    #[test]
    fn null_reads_as_unset() {
        let parsed: GraphicsConfig =
            serde_json::from_str(r#"{"shadows":{"map_size":null}}"#).unwrap();
        assert!(parsed.shadows.is_unset());
    }

    #[test]
    fn unknown_keys_are_rejected_at_both_levels() {
        assert!(serde_json::from_str::<GraphicsConfig>(r#"{"shadow":{}}"#).is_err());
        assert!(serde_json::from_str::<GraphicsConfig>(r#"{"shadows":{"size":1}}"#).is_err());
    }

    #[test]
    fn serializing_writes_only_what_is_set() {
        let config = GraphicsConfig {
            shadows: ShadowsOverrides {
                cascades: Some(4),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(text, r#"{"shadows":{"cascades":4}}"#);
        assert_eq!(
            serde_json::to_string(&GraphicsConfig::default()).unwrap(),
            "{}"
        );
        let back: GraphicsConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn the_binary_form_round_trips_every_section() {
        let config = GraphicsConfig {
            shadows: ShadowsOverrides {
                cascades: Some(2),
                ..Default::default()
            },
            ..Default::default()
        };
        let bytes = postcard::to_allocvec(&config).unwrap();
        assert_eq!(
            postcard::from_bytes::<GraphicsConfig>(&bytes).unwrap(),
            config
        );
    }

    #[test]
    fn a_set_field_replaces_the_base_and_an_unset_one_keeps_it() {
        let mut shadows = Shadows {
            distance: 300,
            ..Default::default()
        };
        ShadowsOverrides {
            map_size: Some(1024),
            update: Some(ShadowUpdate::Hybrid),
            ..Default::default()
        }
        .apply_to(&mut shadows);
        assert_eq!(shadows.map_size, 1024);
        assert_eq!(shadows.update, ShadowUpdate::Hybrid);
        assert_eq!(shadows.distance, 300);
        assert_eq!(shadows.cascades, Shadows::default().cascades);
    }

    #[test]
    fn a_full_section_converts_to_every_field_set() {
        let all = ShadowsOverrides::from(Shadows::default());
        assert_eq!(all.map_size, Some(4096));
        assert!(!all.is_unset());
        let mut base = Shadows {
            map_size: 1,
            distance: 1,
            cascades: 1,
            update: ShadowUpdate::Hybrid,
        };
        all.apply_to(&mut base);
        assert_eq!(base.map_size, 4096);
        assert_eq!(base.update, ShadowUpdate::EveryFrame);
    }

    #[test]
    fn a_set_value_clamps_under_its_section_and_an_unset_one_is_skipped() {
        let mut config = GraphicsConfig {
            shadows: ShadowsOverrides {
                cascades: Some(9),
                ..Default::default()
            },
            anti_aliasing: AntiAliasingOverrides {
                mode: Some(AaMode::Fxaa),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut clamped = Vec::new();
        config.clamp_params(&mut clamped);
        assert_eq!(config.shadows.cascades, Some(4));
        assert_eq!(
            clamped,
            [ClampedParam {
                section: Some("shadows"),
                key: "cascades",
                from: 9.0,
                to: 4.0,
            }]
        );
        assert_eq!(clamped[0].path(), "shadows.cascades");
        let mut none = Vec::new();
        GraphicsConfig::default().clamp_params(&mut none);
        assert!(none.is_empty());
    }

    #[test]
    fn explicit_defaults_set_every_field_of_every_section() {
        let all = GraphicsConfig::explicit_defaults();
        assert!(!all.shadows.is_unset() && !all.culling.is_unset());
        assert_eq!(all.shadows.map_size, Some(4096));
        let value = serde_json::to_value(&all).unwrap();
        assert_eq!(value.as_object().unwrap().len(), GRAPHICS_SECTIONS.len());
    }

    #[test]
    fn section_parameters_are_listed_under_dotted_paths() {
        let params = GraphicsConfig::field_table().params;
        let paths: Vec<_> = params.iter().map(|p| p.path.as_str()).collect();
        assert!(paths.contains(&"shadows.map_size"), "{paths:?}");
        assert!(paths.contains(&"frame_pacing.fps_cap"), "{paths:?}");
        assert!(paths.contains(&"culling.two_pass_occlusion"), "{paths:?}");
        assert_eq!(ShadowsOverrides::PARAMS, Shadows::PARAMS);
    }
}

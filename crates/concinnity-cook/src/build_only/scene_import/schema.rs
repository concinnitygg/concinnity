//! Scene-import schema.

/// Imports a 3D scene file as a single declaration.
///
/// One `SceneImport` stands in for the whole asset graph a scene file
/// describes: its [Texture](#texture)s, [Material](#material)s,
/// [Mesh](#mesh)es, [Model](#model)s, and [Prop](#prop)s. The build expands the
/// import into those concrete assets, so `world.jsonl` stays small and
/// human-editable while the full graph lives in the lock file and compiled
/// blob. Geometry and texture pixels are never inlined into `world.jsonl`.
///
/// Supported `source` formats: `.fbx` and `.glb`.
///
/// **Generated names** are prefixed with the import's own asset `name`
/// (`<name>_mat_0`, `<name>_prim_0`, `<name>_model_0`, ...), so they never
/// clash with hand-authored assets. Because they only appear in the lock file
/// and blob, you never reference them by hand.
///
/// **Camera:** the import frames a [Camera3D](#camera3d) to the scene's bounds
/// so a freshly imported scene is immediately viewable. It is suppressed when
/// the world already declares a `Camera3D` (yours wins) or when `emit_camera`
/// is set to `false`.
///
/// ```rust
/// # use concinnity_cook::authoring::registry::build_only::SceneImport;
/// SceneImport {
///     source: "assets/Bistro/BistroExterior.fbx".into(),
///     texture_max_size: 512,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SceneImport {
    /// Path to the scene file, relative to the project root. `.fbx` or `.glb`.
    pub source: String,
    /// Ceiling on the longest edge of each imported texture, in pixels. Large
    /// source maps (2K-4K) are box-filtered down so the compiled scene, which
    /// stores uncompressed pixels, stays within a sane memory budget. `0` keeps
    /// each texture at its source resolution.
    pub texture_max_size: u32,
    /// Emissive factor applied to a material that carries an emissive map. Scene
    /// files often ship a zero emissive factor that would cancel the map, so a
    /// textured emissive gets this punchy factor instead.
    pub emissive_map_strength: f32,
    /// Whether to emit a [Camera3D](#camera3d) framed to the scene's bounds.
    /// Suppressed automatically when the world already declares a `Camera3D`.
    pub emit_camera: bool,
}

impl Default for SceneImport {
    fn default() -> Self {
        Self {
            source: String::new(),
            texture_max_size: 512,
            emissive_map_strength: 3.0,
            emit_camera: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_import_emits_a_camera_so_the_scene_is_viewable_immediately() {
        // `cn add foo.glb` writes one SceneImport line, so the expansion has to
        // produce something navigable without any further authoring.
        let s = SceneImport::default();
        assert!(s.emit_camera);
        assert_eq!(s.texture_max_size, 512);
        assert_eq!(s.emissive_map_strength, 3.0);
        assert!(s.source.is_empty());
    }

    #[test]
    fn an_authored_import_parses_and_round_trips_through_postcard() {
        let s: SceneImport = serde_json::from_str(
            r#"{"source":"bistro.fbx","texture_max_size":2048,
                "emissive_map_strength":1.0,"emit_camera":false}"#,
        )
        .unwrap();
        assert_eq!(s.source, "bistro.fbx");
        assert!(!s.emit_camera);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: SceneImport = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.texture_max_size, 2048);
        assert_eq!(back.emissive_map_strength, 1.0);
        assert!(!back.emit_camera);
    }
}

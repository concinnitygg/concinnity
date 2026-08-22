// World rendering configuration schema.

/// How often each cascaded-shadow-map slice is re-rendered. The shadow pass
/// re-rasterizes all scene geometry into every cascade, so it is one of the
/// heavier passes; updating distant cascades less often cuts that cost.
///
/// `hybrid` (the default) re-renders the nearest cascade every frame (so close
/// shadows stay crisp) and rotates through the farther cascades one per frame.
/// Distant shadows then lag a few frames while the camera moves, which is
/// imperceptible at that range. `every_frame` re-renders all cascades every
/// frame: pick it for scenes with fast-moving shadow casters where even distant
/// shadow lag is unacceptable. Each cascade is always primed (rendered once)
/// before it is sampled, so there is never missing shadow data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ShadowUpdate {
    /// Re-render every cascade every frame.
    EveryFrame,
    /// Re-render the near cascades every frame and the distant ones on a
    /// rotation.
    #[default]
    Hybrid,
}

/// Rendering settings for the world: frame pacing, shadows, and clear colour.
/// One per world. The GPU backend is chosen by the engine for the platform and
/// is not user-configurable.
///
/// ```json
/// {
///   "name": "gfx",
///   "type": "GraphicsConfig",
///   "args": { "clear_color": [0.1, 0.1, 0.15, 1.0], "frames_in_flight": 2 }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GraphicsConfig {
    /// Cap the render loop at this many frames, then exit. Unset runs until the
    /// window is closed.
    pub max_frames: Option<u64>,
    /// Preferred number of frames in flight (1-3). Higher can smooth pacing at
    /// the cost of input latency.
    pub frames_in_flight: u32,
    /// Cap the frame rate to the display refresh (vsync). Defaults to `false`:
    /// the render loop runs uncapped (DirectX presents with tearing allowed,
    /// Vulkan uses a mailbox present mode), which is what a benchmark wants. Set
    /// to `true` to lock presentation to the monitor refresh, eliminating tearing
    /// and the wasted frames that never reach the screen.
    pub vsync: bool,
    /// Cap the frame rate to this many frames per second. `0` (default) leaves
    /// the loop uncapped. The cap is a CPU-side frame pacer, so it composes with
    /// `vsync`: the more restrictive of the two wins. Useful for limiting heat,
    /// fan noise, and power draw, or matching a fixed refresh.
    pub fps_cap: u32,
    /// Background clear colour [r, g, b, a] in linear 0..1 space.
    pub clear_color: [f32; 4],
    /// Rotation speed of the demo object in radians per second. Only used when
    /// no camera is present.
    pub rotation_speed: f32,
    /// Shadow map resolution in texels (e.g. 2048). Set to 0 to disable shadows.
    pub shadow_map_size: u32,
    /// How often shadow cascades are re-rendered. `hybrid` (default) amortizes
    /// the far cascades across frames; `every_frame` refreshes them all every
    /// frame.
    pub shadow_update: ShadowUpdate,
    /// How far from the camera shadows are cast, in world units (e.g. 80). The
    /// cascades cover from the near plane out to this distance; a larger value
    /// shadows more of the scene but spreads the same shadow-map resolution over
    /// more area (softer, blockier shadows). Capped at the camera far plane.
    pub shadow_distance: u32,
    /// Number of shadow cascades, 1 to 4 (`4` is the default and the maximum).
    /// More cascades keep distant shadows sharper by splitting the view range
    /// into finer slices, at the cost of an extra shadow-map render per cascade;
    /// fewer is cheaper but blockier far from the camera. The slice count covers
    /// the same `shadow_distance` regardless.
    pub shadow_cascades: u32,
    /// Maximum anisotropic-filtering degree for the scene texture sampler
    /// (albedo + normal maps), e.g. 8. Higher keeps textures viewed at a grazing
    /// angle (floors, walls receding into the distance) sharp instead of blurring
    /// along the minor axis, at a small sampling cost. `1` disables anisotropy
    /// (plain trilinear). Clamped to the GPU's supported range (1..16) at init.
    pub anisotropy: u32,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            max_frames: None,
            frames_in_flight: 2,
            vsync: false,
            fps_cap: 0,
            clear_color: [0.01, 0.01, 0.02, 1.0],
            rotation_speed: 1.0,
            shadow_map_size: 2048,
            shadow_update: ShadowUpdate::default(),
            shadow_distance: 80,
            shadow_cascades: 4,
            anisotropy: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_run_uncapped_with_hybrid_cascaded_shadows() {
        let g = GraphicsConfig::default();
        // No frame ceiling and no fps cap: `cn run` renders until it is closed.
        assert_eq!(g.max_frames, None);
        assert_eq!(g.fps_cap, 0);
        assert!(!g.vsync);
        assert_eq!(g.frames_in_flight, 2);
        assert_eq!(g.shadow_update, ShadowUpdate::Hybrid);
        assert_eq!(g.shadow_map_size, 2048);
        assert_eq!(g.shadow_cascades, 4);
        assert_eq!(g.shadow_distance, 80);
        assert_eq!(g.anisotropy, 8);
        assert_eq!(ShadowUpdate::default(), ShadowUpdate::Hybrid);
    }

    #[test]
    fn shadow_update_names_parse_in_snake_case() {
        assert_eq!(
            serde_json::from_str::<ShadowUpdate>(r#""every_frame""#).unwrap(),
            ShadowUpdate::EveryFrame
        );
        assert_eq!(
            serde_json::from_str::<ShadowUpdate>(r#""hybrid""#).unwrap(),
            ShadowUpdate::Hybrid
        );
        assert_eq!(
            serde_json::to_string(&ShadowUpdate::EveryFrame).unwrap(),
            r#""every_frame""#
        );
    }

    #[test]
    fn an_authored_config_parses_and_round_trips_through_postcard() {
        let g: GraphicsConfig = serde_json::from_str(
            r#"{"max_frames":120,"vsync":true,"fps_cap":60,"clear_color":[0,0,0,1],
                "shadow_update":"every_frame","shadow_map_size":4096,"shadow_cascades":2,
                "anisotropy":16}"#,
        )
        .unwrap();
        assert_eq!(g.max_frames, Some(120));
        assert!(g.vsync);
        assert_eq!(g.shadow_update, ShadowUpdate::EveryFrame);

        let bytes = postcard::to_allocvec(&g).unwrap();
        let back: GraphicsConfig = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.max_frames, Some(120));
        assert_eq!(back.fps_cap, 60);
        assert_eq!(back.clear_color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(back.shadow_map_size, 4096);
        assert_eq!(back.shadow_cascades, 2);
        assert_eq!(back.anisotropy, 16);
        // Rotation speed was not authored, so it keeps the schema default.
        assert_eq!(back.rotation_speed, 1.0);
    }
}

//! Engine-injected default opt-out schema.

/// Opts a world out of individual engine-injected defaults.
///
/// A world is completed at build time with standard assets it does not declare
/// itself: the [DebugHud](#debughud) with its chip [TextLabel](#textlabel)s
/// and font, the [StatHud](#stathud) and its chips when the world declares a
/// [MainMenu](#mainmenu), the [PhysicsConfig](#physicsconfig) a world with
/// physics content simulates on, and, when an
/// [EnvironmentMap](#environmentmap) is present, the sky mesh that displays
/// it. Declaring the same asset yourself replaces the injected one; declaring
/// `EngineDefaults` with a flag set to `false` removes it entirely.
///
/// The build records every injected asset in `world-lock.json`; copy an entry
/// from there (or from `cn explain <name>`) into `world.jsonl` to override it.
///
/// ```rust
/// # use concinnity_world::registry::build_only::EngineDefaults;
/// EngineDefaults {
///     debug_hud: false,
///     sky: false,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineDefaults {
    /// Inject the [StatHud](#stathud) with its chip labels and font when the
    /// world declares a [MainMenu](#mainmenu) but no `StatHud`.
    pub hud: bool,
    /// Inject the [DebugHud](#debughud) with its chip labels when the world
    /// declares no `DebugHud`.
    pub debug_hud: bool,
    /// Inject the sky mesh (a skybox [ProceduralMesh](#proceduralmesh),
    /// [Material](#material), and [Prop](#prop)) when the world has an
    /// [EnvironmentMap](#environmentmap) but no skybox mesh. Disable to use an
    /// `EnvironmentMap` for image-based lighting only, with the background
    /// left to `clear_color` or your own geometry.
    pub sky: bool,
    /// Inject an Escape-toggled pause [MainMenu](#mainmenu) when the world
    /// plays a [Story](#story) but declares no `MainMenu`: Resume, Save, Load,
    /// a trimmed Settings screen, and Quit to the story's title. Disable to
    /// leave a story with no pause menu, or declare your own `MainMenu` to
    /// replace it.
    pub story_pause_menu: bool,
    /// Inject the [LoadingOverlay](#loadingoverlay) with its screen, backdrop,
    /// progress bar, and label when the world declares [Scene](#scene)s and a
    /// [StreamingConfig](#streamingconfig) but no `LoadingOverlay`. Disable to
    /// jump between scenes with no loading screen while their content streams
    /// in.
    pub loading_overlay: bool,
    /// Inject a [PhysicsConfig](#physicsconfig) with the engine's own values
    /// when the world has physics content -- a [RigidBody](#rigidbody), a
    /// [PropBody](#propbody), a [TriggerVolume](#triggervolume), or a
    /// [SkinnedMesh](#skinnedmesh) with a `capsule` -- but declares no
    /// `PhysicsConfig`. Physics runs on those values either way; the injected
    /// asset is what makes them visible in `world-lock.json` and editable,
    /// `spawn_headroom` above all. Disable to leave them implicit.
    pub physics_config: bool,
}

impl Default for EngineDefaults {
    fn default() -> Self {
        Self {
            hud: true,
            debug_hud: true,
            sky: true,
            story_pause_menu: true,
            loading_overlay: true,
            physics_config: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_injected_default_is_on_until_opted_out_of() {
        // This asset exists only to turn injection off, so declaring it without
        // saying which one must change nothing.
        let d = EngineDefaults::default();
        assert!(d.hud);
        assert!(d.debug_hud);
        assert!(d.sky);
        assert!(d.story_pause_menu);
        assert!(d.loading_overlay);
        assert!(d.physics_config);

        let declared: EngineDefaults = serde_json::from_str("{}").unwrap();
        assert!(declared.hud && declared.debug_hud && declared.sky);
        assert!(declared.story_pause_menu && declared.loading_overlay);
        assert!(declared.physics_config);
    }

    #[test]
    fn opting_out_of_one_default_leaves_the_rest_alone() {
        let d: EngineDefaults = serde_json::from_str(r#"{"sky":false}"#).unwrap();
        assert!(!d.sky);
        assert!(d.hud && d.debug_hud && d.story_pause_menu && d.loading_overlay);
        assert!(d.physics_config);

        let bytes = postcard::to_allocvec(&d).unwrap();
        let back: EngineDefaults = postcard::from_bytes(&bytes).unwrap();
        assert!(!back.sky);
        assert!(back.hud);
    }
}

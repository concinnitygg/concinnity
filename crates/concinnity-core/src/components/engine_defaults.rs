// Engine-injected default opt-out schema.

/// Opts a world out of individual engine-injected defaults.
///
/// A world is completed at start with the standard components it does not
/// declare itself: the [DebugHud](#debughud) with its chip
/// [TextLabel](#textlabel)s and font, the chips and font of a
/// [StatHud](#stathud), the [PhysicsConfig](#physicsconfig) a world with
/// physics content simulates on, the [LoadingOverlay](#loadingoverlay) a
/// streamed world waits behind, and, when an
/// [EnvironmentMap](#environmentmap) is present, the sky mesh that displays
/// it. Declaring a piece yourself keeps the injection from filling that slot;
/// declaring `EngineDefaults` with a flag set to `false` removes the whole
/// default.
///
/// Two defaults are stated in terms of a [MainMenu](#mainmenu), which the
/// build expands away, so they are injected by the build instead: the
/// `StatHud` a menu world drives, and a story world's pause menu.
///
/// ```rust
/// # use concinnity_core::components::EngineDefaults;
/// EngineDefaults {
///     debug_hud: false,
///     sky: false,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct EngineDefaults {
    /// Inject the [StatHud](#stathud) when the world declares a
    /// [MainMenu](#mainmenu) but no `StatHud`, and fill any declared
    /// `StatHud`'s unset chip labels with chips.
    #[asset(default = true)]
    pub hud: bool,
    /// Inject the [DebugHud](#debughud) with its chip labels when the world
    /// declares no `DebugHud`.
    #[asset(default = true)]
    pub debug_hud: bool,
    /// Inject the sky mesh (a skybox mesh, [Material](#material), and
    /// [Prop](#prop)) when the world has an
    /// [EnvironmentMap](#environmentmap) but no skybox mesh. Disable to use an
    /// `EnvironmentMap` for image-based lighting only, with the background
    /// left to `clear_color` or your own geometry.
    #[asset(default = true)]
    pub sky: bool,
    /// Inject an Escape-toggled pause [MainMenu](#mainmenu) when the world
    /// plays a [Story](#story) but declares no `MainMenu`: Resume, Save, Load,
    /// a trimmed Settings screen, and Quit to the story's title. Disable to
    /// leave a story with no pause menu, or declare your own `MainMenu` to
    /// replace it.
    #[asset(default = true)]
    pub story_pause_menu: bool,
    /// Inject the [LoadingOverlay](#loadingoverlay) with its screen, backdrop,
    /// progress bar, and label when the world declares [Scene](#scene)s and a
    /// [StreamingConfig](#streamingconfig) but no `LoadingOverlay`. Disable to
    /// jump between scenes with no loading screen while their content streams
    /// in.
    #[asset(default = true)]
    pub loading_overlay: bool,
    /// Inject a [PhysicsConfig](#physicsconfig) with the engine's own values
    /// when the world has physics content -- a [RigidBody](#rigidbody), a
    /// [PropBody](#propbody), a [TriggerVolume](#triggervolume), or a
    /// [SkinnedMesh](#skinnedmesh) with a `capsule` -- but declares no
    /// `PhysicsConfig`. Physics runs on those values either way; the injected
    /// component is what makes them visible to tooling. Disable to leave them
    /// implicit.
    #[asset(default = true)]
    pub physics_config: bool,
}

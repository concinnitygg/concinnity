// Scene-loading progress overlay schema.

use crate::components::{Screen, Sprite, TextLabel};
use crate::ecs::Ref;

/// Requests the scene-loading overlay: a full-window backdrop with a progress
/// bar, shown while a scene jump waits for its streamed content and faded out
/// once the destination scene is fully resident.
///
/// The overlay is assembled from ordinary UI elements the fields reference: a
/// [Screen](#screen) that hosts them, a backdrop [Sprite](#sprite) covering
/// the canvas, a progress-bar track and fill [Sprite](#sprite) pair, and a
/// [TextLabel](#textlabel) above the bar. Each frame the engine widens the
/// fill to the destination scene's load progress and rewrites the label with
/// the percentage; restyle any piece by declaring it yourself and pointing the
/// overlay's field at it.
///
/// While the overlay's screen is up the world pauses and its render is
/// skipped, exactly like an opaque menu, but streaming keeps running so the
/// load it reports can finish. Scenes whose content is already resident jump
/// without the overlay ever appearing.
///
/// Every rendering world that declares [Scene](#scene)s and a
/// [StreamingConfig](#streamingconfig) receives a `LoadingOverlay` at start
/// when it declares none, and any field left unset receives the piece it
/// names, so the example below is only needed to restyle them. Declare an
/// [EngineDefaults](#enginedefaults) with `"loading_overlay": false` to leave
/// the world without one.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct LoadingOverlay {
    /// [Screen](#screen) the overlay shows while a scene loads. Its
    /// `pauses_world` (on by default) freezes the world beneath the overlay so
    /// the destination scene starts fresh when revealed.
    pub screen: Option<Ref<Screen>>,
    /// Backdrop [Sprite](#sprite) covering the canvas. Its tint alpha is
    /// animated to reveal the scene once loading completes; an opaque tint
    /// hides the still-loading world completely.
    pub backdrop: Option<Ref<Sprite>>,
    /// Progress-bar track [Sprite](#sprite); its width is the bar's full
    /// extent the fill is measured against.
    pub track: Option<Ref<Sprite>>,
    /// Progress-bar fill [Sprite](#sprite); the engine sets its width to the
    /// track width times the destination scene's load progress each frame.
    pub fill: Option<Ref<Sprite>>,
    /// [TextLabel](#textlabel) rewritten each frame with the load percentage.
    pub label: Option<Ref<TextLabel>>,
}

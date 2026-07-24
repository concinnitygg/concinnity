// Scene-loading progress overlay schema.

use crate::{AssetId, de_opt_asset_ref};

/// Requests the scene-loading overlay: a full-window backdrop with a progress
/// bar, shown while a scene jump waits for its streamed content and faded out
/// once the destination scene is fully resident.
///
/// The overlay is assembled from ordinary UI elements the fields reference: a
/// [Screen](#screen) that hosts them, a backdrop [Sprite](#sprite) covering
/// the canvas, a progress-bar track and fill [Sprite](#sprite) pair, and a
/// [TextLabel](#textlabel) above the bar. Each frame the engine widens the
/// fill to the destination scene's load progress and rewrites the label with
/// the percentage; restyle any piece by declaring it yourself under the same
/// name.
///
/// While the overlay's screen is up the world pauses and its render is
/// skipped, exactly like an opaque menu, but streaming keeps running so the
/// load it reports can finish. Scenes whose content is already resident jump
/// without the overlay ever appearing.
///
/// Every rendering world that declares [Scene](#scene)s and a
/// [StreamingConfig](#streamingconfig) receives a `LoadingOverlay` and its
/// pieces at build time when it declares none, so the example below is only
/// needed to restyle them. Declare an [EngineDefaults](#enginedefaults) with
/// `"loading_overlay": false` to remove it from the build entirely.
///
/// ```jsonl
/// {"type":"Screen","name":"loading_screen","args":{"fade_in_secs":0.15}}
/// {"type":"Sprite","name":"loading_screen_backdrop","args":{"x":0,"y":0,"width":1280,"height":720,"tint":[0,0,0,1],"fit":"cover"}}
/// {"type":"Sprite","name":"loading_screen_track","args":{"x":400,"y":600,"width":480,"height":8,"tint":[0.25,0.25,0.25,1],"corner_radius":4}}
/// {"type":"Sprite","name":"loading_screen_fill","args":{"x":400,"y":600,"width":0,"height":8,"tint":[0.92,0.92,0.92,1],"corner_radius":4,"visible":false}}
/// {"type":"TextLabel","name":"loading_screen_label","args":{"font":"hud_font","content":"Loading","x":640,"y":566,"align":"center"}}
/// {"type":"LoadingOverlay","name":"loading_overlay","args":{"screen":"loading_screen","backdrop":"loading_screen_backdrop","track":"loading_screen_track","fill":"loading_screen_fill","label":"loading_screen_label"}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LoadingOverlay {
    /// [Screen](#screen) the overlay shows while a scene loads. Its
    /// `pauses_world` (on by default) freezes the world beneath the overlay so
    /// the destination scene starts fresh when revealed.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
    /// Backdrop [Sprite](#sprite) covering the canvas. Its tint alpha is
    /// animated to reveal the scene once loading completes; an opaque tint
    /// hides the still-loading world completely.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub backdrop: Option<AssetId>,
    /// Progress-bar track [Sprite](#sprite); its width is the bar's full
    /// extent the fill is measured against.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub track: Option<AssetId>,
    /// Progress-bar fill [Sprite](#sprite); the engine sets its width to the
    /// track width times the destination scene's load progress each frame.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub fill: Option<AssetId>,
    /// [TextLabel](#textlabel) rewritten each frame with the load percentage.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub label: Option<AssetId>,
}

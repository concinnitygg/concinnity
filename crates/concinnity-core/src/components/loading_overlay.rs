// Scene-loading progress overlay schema.

use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_overlay_claims_no_pieces() {
        let o = LoadingOverlay::default();
        assert!(o.screen.is_none());
        assert!(o.backdrop.is_none());
        assert!(o.track.is_none());
        assert!(o.fill.is_none());
        assert!(o.label.is_none());
    }

    #[test]
    fn each_piece_binds_its_own_asset_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let o: LoadingOverlay = serde_json::from_str(
            r#"{"screen":"load","backdrop":"dim","track":"bar_bg","fill":"bar","label":"pct"}"#,
        )
        .unwrap();
        assert_eq!(o.screen, Some(AssetId(4)));
        assert_eq!(o.backdrop, Some(AssetId(3)));
        assert_eq!(o.track, Some(AssetId(6)));
        assert_eq!(o.fill, Some(AssetId(3)));
        assert_eq!(o.label, Some(AssetId(3)));

        let bytes = postcard::to_allocvec(&o).unwrap();
        let back: LoadingOverlay = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.track, Some(AssetId(6)));
        assert_eq!(back.screen, Some(AssetId(4)));
    }
}

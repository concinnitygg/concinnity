//! Settings-slider row schema.

/// A settings row that sets a continuous value by dragging a handle along a
/// track.
///
/// `Slider` is a build-time shorthand for one row of a settings menu: a
/// left-aligned name, a draggable track with a handle, and a right-aligned
/// current value. It expands into a [TextLabel](#textlabel) for the name, a
/// [TextLabel](#textlabel) for the value, two [Sprite](#sprite)s (the track and
/// the handle), and a [HitRegion](#hitregion) covering the track that fires a
/// `"setting:<setting>:drag"` action. While the region is pressed the handle
/// follows the cursor and the value updates live.
///
/// The `setting` field names an engine setting the runtime knows how to map
/// from a fraction, apply, and format (e.g. `"exposure"`); its value range and
/// display format live in the engine, not here. The value label and handle
/// show a placeholder position at build time and are corrected to the live
/// value when the world starts.
///
/// Generated names are prefixed with this asset's `name` (`<name>_label`,
/// `<name>_value`, `<name>_track`, `<name>_handle`, `<name>_drag`), so they
/// never clash with hand-authored assets.
///
/// ```rust
/// # use concinnity_world::registry::build_only::Slider;
/// Slider {
///     setting: "exposure".into(),
///     label: "Exposure".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Slider {
    /// Engine setting this row controls (e.g. `"exposure"`). Must be a setting
    /// the runtime recognises as a slider; an unknown key renders but does
    /// nothing on drag.
    pub setting: String,
    /// Display name shown at the left of the row.
    pub label: String,
    /// Left edge of the row in window pixels.
    pub x: f32,
    /// Top edge of the row in window pixels.
    pub y: f32,
    /// Row width in window pixels (name sits at the left, track and value at
    /// the right).
    pub width: f32,
    /// Row height in window pixels (the draggable region's height).
    pub height: f32,
    /// [Font](#font) for the row text. Empty uses the built-in font.
    pub font: String,
    /// Pixel size of the row text when it uses the built-in font (that is, when
    /// `font` is empty). Ignored when `font` names a [Font](#font), which
    /// carries its own size.
    pub font_px: f32,
    /// Linear-space RGB color of the name text.
    pub text_color: [f32; 3],
    /// Linear-space RGB color of the value text.
    pub value_color: [f32; 3],
    /// Scale applied to the row text.
    pub text_scale: f32,
    /// RGBA color of the track bar behind the handle.
    pub track_color: [f32; 4],
    /// RGBA color of the draggable handle.
    pub handle_color: [f32; 4],
}

impl Default for Slider {
    fn default() -> Self {
        Self {
            setting: String::new(),
            label: String::new(),
            x: 0.0,
            y: 0.0,
            width: 360.0,
            height: 48.0,
            font: String::new(),
            font_px: 48.0,
            text_color: [0.85, 0.85, 0.85],
            value_color: [0.85, 0.85, 0.85],
            text_scale: 1.0,
            track_color: [0.28, 0.28, 0.32, 1.0],
            handle_color: [1.0, 0.85, 0.3, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_slider_matches_the_settings_row_it_sits_in() {
        let s = Slider::default();
        assert!(s.setting.is_empty());
        assert!(s.label.is_empty());
        assert_eq!((s.width, s.height), (360.0, 48.0));
        assert_eq!(s.text_scale, 1.0);
        // The handle is opaque against a dimmer track so the value reads at a
        // glance.
        assert_eq!(s.track_color[3], 1.0);
        assert_eq!(s.handle_color, [1.0, 0.85, 0.3, 1.0]);
    }

    #[test]
    fn an_authored_slider_parses_and_round_trips_through_postcard() {
        let s: Slider = serde_json::from_str(
            r#"{"setting":"master_volume","label":"Volume","x":40,"y":200,"width":420,
                "height":40,"font":"body","font_px":32,"text_color":[1,1,1],
                "value_color":[0.7,0.7,0.7],"text_scale":1.1,
                "track_color":[0,0,0,1],"handle_color":[1,1,1,1]}"#,
        )
        .unwrap();
        assert_eq!(s.setting, "master_volume");

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Slider = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.label, "Volume");
        assert_eq!((back.x, back.y), (40.0, 200.0));
        assert_eq!((back.width, back.height), (420.0, 40.0));
        assert_eq!(back.font, "body");
        assert_eq!(back.font_px, 32.0);
        assert_eq!(back.value_color, [0.7, 0.7, 0.7]);
        assert_eq!(back.text_scale, 1.1);
        assert_eq!(back.track_color, [0.0, 0.0, 0.0, 1.0]);
    }
}

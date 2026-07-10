// src/gfx/overlay.rs
//
// Screen-overlay scaling math shared by the build pipeline (which lays menus
// out against a fixed reference canvas) and the client renderer (which scales
// that canvas to the live window). View-owned UI (menus, settings) is authored
// in a fixed reference resolution; at runtime the whole overlay is uniformly
// scaled to fit the window, preserving aspect and staying centered, so a menu
// looks the same proportion of the screen at any window size.

// Reference resolution menus are authored against. Window-pixel coordinates of
// view-owned UI are interpreted in this space and scaled to the live window.
pub const UI_REFERENCE_SIZE: [f32; 2] = [1280.0, 720.0];

// A uniform similarity transform mapping the reference canvas to the live
// window: a single scale plus a recentering. Built from the live viewport; an
// invalid (zero) viewport yields the identity (overlay drawn at reference
// pixels), which is what unit tests and the pre-backend init frames see.
#[derive(Debug, Clone, Copy)]
pub struct OverlayTransform {
    scale: f32,
    // Window-space center the reference center maps to.
    screen_cx: f32,
    screen_cy: f32,
    // Reference-space center (half the reference size).
    ref_cx: f32,
    ref_cy: f32,
}

impl OverlayTransform {
    // Build the transform for a live logical viewport `[width, height]`. A
    // degenerate viewport gives the identity transform.
    pub fn from_viewport(viewport: [f32; 2]) -> Self {
        let [rw, rh] = UI_REFERENCE_SIZE;
        let ref_cx = rw / 2.0;
        let ref_cy = rh / 2.0;
        let [vw, vh] = viewport;
        if vw <= 0.0 || vh <= 0.0 {
            return Self {
                scale: 1.0,
                screen_cx: ref_cx,
                screen_cy: ref_cy,
                ref_cx,
                ref_cy,
            };
        }
        // Uniform "fit": the smaller axis ratio, so the reference canvas always
        // fits inside the window without distorting text.
        let scale = (vw / rw).min(vh / rh);
        Self {
            scale,
            screen_cx: vw / 2.0,
            screen_cy: vh / 2.0,
            ref_cx,
            ref_cy,
        }
    }

    // Build the "cover" transform for a live logical viewport: the larger axis
    // ratio, so the reference canvas always fills the window (the overflowing
    // axis is cropped equally on both sides). Used by full-bleed stage imagery
    // (scene backdrops, character portraits) that must reach the window edges
    // without distorting; the canvas bottom maps at or below the window
    // bottom, so bottom-anchored content stays flush at any aspect ratio.
    pub fn cover_from_viewport(viewport: [f32; 2]) -> Self {
        let mut t = Self::from_viewport(viewport);
        let [rw, rh] = UI_REFERENCE_SIZE;
        let [vw, vh] = viewport;
        if vw > 0.0 && vh > 0.0 {
            t.scale = (vw / rw).max(vh / rh);
        }
        t
    }

    // Build the "bottom-anchored" transform for a live logical viewport: the
    // `fit` scale (no cropping, elements keep their proportions), but shifted
    // vertically so the reference bottom edge (y = reference height) maps to the
    // window bottom. Bottom-anchored overlay furniture (a dialog box and its
    // controls) hugs the window bottom at any aspect ratio, where a plain `fit`
    // would float it above the letterbox margin.
    pub fn bottom_anchored_from_viewport(viewport: [f32; 2]) -> Self {
        let mut t = Self::from_viewport(viewport);
        let [_, rh] = UI_REFERENCE_SIZE;
        let [_, vh] = viewport;
        if vh > 0.0 {
            // forward(_, rh).1 == vh  <=>  screen_cy = vh - (rh - ref_cy) * scale
            t.screen_cy = vh - (rh - t.ref_cy) * t.scale;
        }
        t
    }

    // The uniform scale factor applied to sizes (glyph scale, sprite extent).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    // Map a reference-space point to window space.
    pub fn forward(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.screen_cx + (x - self.ref_cx) * self.scale,
            self.screen_cy + (y - self.ref_cy) * self.scale,
        )
    }

    // Map a window-space point back to reference space (the inverse of
    // `forward`). Used to hit-test the live cursor against reference-space UI
    // rects.
    pub fn inverse(&self, x: f32, y: f32) -> (f32, f32) {
        let s = if self.scale != 0.0 { self.scale } else { 1.0 };
        (
            self.ref_cx + (x - self.screen_cx) / s,
            self.ref_cy + (y - self.screen_cy) / s,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_viewport_is_identity() {
        let t = OverlayTransform::from_viewport([0.0, 0.0]);
        assert_eq!(t.scale(), 1.0);
        // A point maps to itself.
        let (x, y) = t.forward(100.0, 200.0);
        assert!((x - 100.0).abs() < 1e-4 && (y - 200.0).abs() < 1e-4);
    }

    #[test]
    fn exact_reference_size_is_unit_scale_and_centered() {
        let t = OverlayTransform::from_viewport(UI_REFERENCE_SIZE);
        assert!((t.scale() - 1.0).abs() < 1e-4);
        // The reference center maps to the window center.
        let (cx, cy) = t.forward(UI_REFERENCE_SIZE[0] / 2.0, UI_REFERENCE_SIZE[1] / 2.0);
        assert!((cx - UI_REFERENCE_SIZE[0] / 2.0).abs() < 1e-4);
        assert!((cy - UI_REFERENCE_SIZE[1] / 2.0).abs() < 1e-4);
    }

    #[test]
    fn doubling_both_axes_doubles_scale() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        let t = OverlayTransform::from_viewport([rw * 2.0, rh * 2.0]);
        assert!((t.scale() - 2.0).abs() < 1e-4);
        // The reference origin maps such that the canvas stays centered: the
        // reference center sits at the window center.
        let (cx, cy) = t.forward(rw / 2.0, rh / 2.0);
        assert!((cx - rw).abs() < 1e-4 && (cy - rh).abs() < 1e-4);
    }

    #[test]
    fn wider_window_fits_to_height_and_letterboxes_width() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        // Twice as wide, same height: the limiting axis is height (ratio 1.0).
        let t = OverlayTransform::from_viewport([rw * 2.0, rh]);
        assert!((t.scale() - 1.0).abs() < 1e-4);
        // The canvas stays centered horizontally: reference left edge (x=0)
        // lands at half a reference width in from the window's left.
        let (x0, _) = t.forward(0.0, 0.0);
        assert!((x0 - rw / 2.0).abs() < 1e-4, "x0={x0}");
    }

    #[test]
    fn cover_uses_the_larger_axis_ratio() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        // A 4:3 window is taller than the 16:9 reference: fit is width-limited,
        // cover is height-limited.
        let t = OverlayTransform::cover_from_viewport([1024.0, 768.0]);
        assert!((t.scale() - 768.0 / rh).abs() < 1e-4);
        // The canvas fills the window vertically: the reference bottom maps
        // exactly to the window bottom (no letterbox bar).
        let (_, by) = t.forward(rw / 2.0, rh);
        assert!((by - 768.0).abs() < 1e-3, "by={by}");
        // The overflowing axis crops equally: the reference left edge maps
        // off-window by half the overflow.
        let scaled_w = rw * t.scale();
        let (x0, _) = t.forward(0.0, 0.0);
        assert!((x0 - (1024.0 - scaled_w) / 2.0).abs() < 1e-3, "x0={x0}");
    }

    #[test]
    fn cover_of_a_degenerate_viewport_is_identity() {
        let t = OverlayTransform::cover_from_viewport([0.0, 0.0]);
        assert_eq!(t.scale(), 1.0);
    }

    #[test]
    fn bottom_anchored_maps_the_reference_bottom_to_the_window_bottom() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        // A window taller than the 16:9 reference: plain fit would leave a
        // margin below the canvas; bottom-anchored pins the canvas bottom to
        // the window bottom while keeping the fit scale.
        let vh = 1450.0;
        let t = OverlayTransform::bottom_anchored_from_viewport([rw * 1.5, vh]);
        assert!((t.scale() - 1.5).abs() < 1e-4, "keeps the fit scale");
        let (_, by) = t.forward(rw / 2.0, rh);
        assert!(
            (by - vh).abs() < 1e-3,
            "reference bottom at window bottom: {by}"
        );
        // Horizontal centering is unchanged from `fit`.
        let (cx, _) = t.forward(rw / 2.0, rh / 2.0);
        assert!((cx - rw * 1.5 / 2.0).abs() < 1e-3, "cx={cx}");
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let t = OverlayTransform::from_viewport([2560.0, 1440.0]);
        let (sx, sy) = t.forward(300.0, 410.0);
        let (rx, ry) = t.inverse(sx, sy);
        assert!((rx - 300.0).abs() < 1e-3, "rx={rx}");
        assert!((ry - 410.0).abs() < 1e-3, "ry={ry}");
    }
}

// A clipped area the app draws shapes into, panned by a drag.

use alloc::string::String;

use crate::components::{Screen, SpriteFit, UiShape};
use crate::ecs::Ref;

use super::WidgetStyle;
use super::scroll_view::ScrollView;

/// An area that draws a named list of shapes, clipped to itself, such as a
/// chart of cards and wires or marks laid over a view. The app writes the
/// shape list into its UI state at runtime.
///
/// The shapes are placed in the canvas's own space and moved by its pan. A
/// press on a shape with a tag sends a `UiEvent` named by `on_press` carrying
/// the tag, or by `on_extend` when Ctrl, Cmd, or Shift is held and it is set.
/// A `pannable` canvas is dragged by a press on its empty space and scrolled
/// by the wheel, never past its shapes, and shows a scrollbar along each edge
/// its shapes reach past. A `passthrough` canvas takes no input at all, so
/// what lies under it does.
///
/// ```rust
/// # use concinnity_engine::components::Canvas;
/// Canvas {
///     shapes: "chart".into(),
///     on_press: "card".into(),
///     pannable: true,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent flags of one canvas, each read on its own"
)]
pub struct Canvas {
    /// The shape list in the app's UI state the canvas draws.
    pub shapes: String,
    /// The event a press on a tagged shape sends.
    pub on_press: String,
    /// The event a press on a tagged shape with Ctrl, Cmd, or Shift held
    /// sends instead, when set.
    pub on_extend: String,
    /// Whether a drag on empty space and the wheel pan the shapes.
    pub pannable: bool,
    /// Whether the canvas takes no input, leaving every press, drag, and wheel
    /// to what lies under it.
    pub passthrough: bool,
    /// Whether shapes are cut off at the canvas's edges.
    #[asset(default = true)]
    pub clip: bool,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    #[asset(default = 200.0)]
    pub width: f32,
    /// Height.
    #[asset(default = 150.0)]
    pub height: f32,
    /// How the canvas draws its text, and its background when that is not
    /// clear.
    pub style: WidgetStyle,
    /// When false the canvas is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned canvas maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the canvas belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime: how far the shapes are panned, `[x, y]`, positive toward the
    /// shapes past the right and bottom edges. The app may set it. Not
    /// authored.
    #[serde(skip)]
    pub pan: [f32; 2],
    /// Runtime: the tag of the shape under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<u32>,
}

impl Canvas {
    /// The canvas's rectangle, `[x, y, width, height]`.
    pub fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Where `shape` draws, panned and placed in the canvas's space.
    pub fn placed(&self, shape: &UiShape) -> [f32; 4] {
        let [x, y, w, h] = shape.rect;
        [self.x + x - self.pan[0], self.y + y - self.pan[1], w, h]
    }

    /// The pan furthest along each axis that still shows `shapes`: their
    /// extent past the canvas's size, or none.
    pub fn max_pan(&self, shapes: &[UiShape]) -> [f32; 2] {
        let mut far = [0.0_f32; 2];
        for s in shapes {
            far[0] = far[0].max(s.rect[0] + s.rect[2]);
            far[1] = far[1].max(s.rect[1] + s.rect[3]);
        }
        [
            (far[0] - self.width).max(0.0),
            (far[1] - self.height).max(0.0),
        ]
    }

    /// `pan` kept within what still shows `shapes`.
    pub fn clamp_pan(&self, pan: [f32; 2], shapes: &[UiShape]) -> [f32; 2] {
        let max = self.max_pan(shapes);
        [pan[0].clamp(0.0, max[0]), pan[1].clamp(0.0, max[1])]
    }

    /// The horizontal and vertical scrollbars of a pannable canvas, each a
    /// track and thumb, along the edges `shapes` reach past.
    pub fn scrollbars(&self, shapes: &[UiShape]) -> [Option<([f32; 4], [f32; 4])>; 2] {
        if !self.pannable {
            return [None, None];
        }
        let max = self.max_pan(shapes);
        let vertical = ScrollView {
            rect: self.rect(),
            content: self.height + max[1],
            scroll: self.pan[1],
        };
        // The horizontal bar is the vertical one of the canvas turned on its side.
        let flip = |r: [f32; 4]| [r[1], r[0], r[3], r[2]];
        let horizontal = ScrollView {
            rect: flip(self.rect()),
            content: self.width + max[0],
            scroll: self.pan[0],
        };
        [
            horizontal
                .scrollbar()
                .map(|(track, thumb)| (flip(track), flip(thumb))),
            vertical.scrollbar(),
        ]
    }

    /// The tag of the frontmost tagged shape under `(x, y)`.
    pub fn tag_at(&self, x: f32, y: f32, shapes: &[UiShape]) -> Option<u32> {
        if !super::point_in(x, y, self.rect()) {
            return None;
        }
        shapes
            .iter()
            .rev()
            .filter(|s| s.tag.is_some())
            .find(|s| super::point_in(x, y, self.placed(s)))
            .and_then(|s| s.tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn shape(rect: [f32; 4], tag: Option<u32>) -> UiShape {
        UiShape {
            rect,
            tag,
            ..UiShape::default()
        }
    }

    fn canvas() -> Canvas {
        Canvas {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
            ..Default::default()
        }
    }

    #[test]
    fn shapes_pan_and_the_frontmost_tagged_one_is_hit() {
        let mut c = canvas();
        let shapes = vec![
            shape([0.0, 0.0, 40.0, 40.0], Some(1)),
            shape([20.0, 0.0, 40.0, 40.0], Some(2)),
            shape([0.0, 0.0, 400.0, 400.0], None),
        ];
        assert_eq!(c.tag_at(35.0, 25.0, &shapes), Some(2), "frontmost tagged");
        assert_eq!(c.tag_at(15.0, 25.0, &shapes), Some(1));
        assert_eq!(c.tag_at(500.0, 25.0, &shapes), None, "outside the canvas");
        c.pan = [20.0, 0.0];
        assert_eq!(c.placed(&shapes[0]), [-10.0, 20.0, 40.0, 40.0]);
        assert_eq!(c.tag_at(15.0, 25.0, &shapes), Some(2));
    }

    #[test]
    fn a_pannable_canvas_shows_a_bar_along_each_overflowing_edge() {
        let mut c = canvas();
        let wide = vec![shape([0.0, 0.0, 300.0, 40.0], None)];
        assert_eq!(c.scrollbars(&wide), [None, None], "not pannable");
        c.pannable = true;
        let [h, v] = c.scrollbars(&wide);
        assert!(v.is_none(), "the shapes fit its height");
        let (track, thumb) = h.unwrap();
        assert_eq!(track[1] + track[3], 70.0, "along the bottom edge");
        assert_eq!(track[2], 100.0);
        assert!((thumb[2] - 100.0 / 3.0).abs() < 1e-3, "a third of it shows");
        c.pan = [200.0, 0.0];
        let (_, end) = c.scrollbars(&wide)[0].unwrap();
        assert!((end[0] + end[2] - 110.0).abs() < 1e-3, "panned to the end");
    }

    #[test]
    fn the_pan_stops_at_the_shapes_extent() {
        let c = canvas();
        let shapes = vec![shape([0.0, 0.0, 300.0, 40.0], None)];
        assert_eq!(c.max_pan(&shapes), [200.0, 0.0]);
        assert_eq!(c.clamp_pan([500.0, 30.0], &shapes), [200.0, 0.0]);
        assert_eq!(c.clamp_pan([-5.0, -5.0], &shapes), [0.0, 0.0]);
    }
}

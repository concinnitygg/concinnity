// The app's UI state: named values and lists the overlay reads and writes, and
// the app's own systems fill.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::components::{TextAlign, UiValue};
use crate::ecs::TextureHandle;

/// One row of a list in the app's [`UiState`]: what a [List](#list) draws for
/// it, and where it sits in a tree.
///
/// A tree is a list whose items carry a `depth`: the app flattens it, listing
/// an expanded item's children right after it one level deeper, and leaves a
/// collapsed item's children out.
#[derive(Debug, Clone, Default, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent flags of one row, each read on its own"
)]
pub struct UiItem {
    /// The row's text.
    pub text: String,
    /// Dimmer text drawn at the row's right end, such as a count or a type.
    pub detail: String,
    /// How many levels deep the item sits in a tree: 0 at the top.
    pub depth: u16,
    /// Whether the item has children to show, so its row draws an expander.
    pub expandable: bool,
    /// Whether its children are listed after it.
    pub expanded: bool,
    /// Whether the row draws dimmed, such as an item hidden or unavailable.
    pub dimmed: bool,
    /// A color the row's text draws in, in place of the list's or menu's, such
    /// as a log line's severity or a destructive menu entry.
    pub color: Option<[f32; 3]>,
    /// A color the detail draws in, in place of the dimmer text color.
    pub detail_color: Option<[f32; 3]>,
    /// Whether the item is one of several selected, for a list or grid whose
    /// app keeps a selection of many items rather than one index.
    pub selected: bool,
    /// An image drawn with the item: at a list row's head, or filling a grid
    /// cell above its text.
    pub image: Option<UiImage>,
    /// Which of a list's toggle columns are on for the row, bit `i` for column
    /// `i`; unset, the row has no toggle cells.
    pub toggles: Option<u8>,
}

/// An image a [`UiItem`] shows, fitted inside the space it is given without
/// stretching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiImage {
    /// The [Texture](#texture) drawn.
    pub texture: TextureHandle,
    /// The texture's size in pixels, whose aspect the image keeps.
    pub size: [u32; 2],
}

impl UiImage {
    /// The rectangle the image fills inside `area`, `[x, y, width, height]`:
    /// as large as fits, centered, at the texture's aspect.
    pub fn fitted(&self, area: [f32; 4]) -> [f32; 4] {
        let [w, h] = self.size;
        if w == 0 || h == 0 {
            return area;
        }
        let scale = (area[2] / w as f32).min(area[3] / h as f32);
        let (fw, fh) = (w as f32 * scale, h as f32 * scale);
        [
            area[0] + (area[2] - fw) * 0.5,
            area[1] + (area[3] - fh) * 0.5,
            fw,
            fh,
        ]
    }
}

/// One shape a [Canvas](#canvas) draws from a list in the app's [`UiState`]:
/// a rectangle in the canvas's own space, filled, outlined, and rounded, with
/// text written inside it. A shape with a `tag` reports a press on it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UiShape {
    /// `[x, y, width, height]` from the canvas's top-left before it pans.
    pub rect: [f32; 4],
    /// RGBA fill; a clear one draws only the outline and the text.
    pub fill: [f32; 4],
    /// RGBA outline color.
    pub border: [f32; 4],
    /// Outline width in pixels; 0 draws none.
    pub border_width: f32,
    /// Corner rounding in pixels.
    pub radius: f32,
    /// Text written on one line inside the rectangle, cut short past its
    /// width.
    pub text: String,
    /// The text's color.
    pub text_color: [f32; 3],
    /// The text's size as a multiple of the canvas's; 0 takes the canvas's.
    pub text_scale: f32,
    /// Where the text sits across the rectangle.
    pub align: TextAlign,
    /// What a press on the shape reports; unset, a press passes through it.
    pub tag: Option<u32>,
}

impl UiShape {
    /// How far a shape's text sits in from its left or right edge.
    pub const TEXT_INSET: f32 = 6.0;
}

impl UiItem {
    /// A plain row showing `text`.
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Self::default()
        }
    }
}

/// The app's UI state: named values the overlay reads and writes through
/// `state:` bindings, and named lists a [List](#list) shows, which the app's
/// own systems read and write directly. Kept apart from the world's variables.
#[derive(Debug, Clone, Default)]
pub struct UiState {
    values: BTreeMap<String, UiValue>,
    lists: BTreeMap<String, Vec<UiItem>>,
    shapes: BTreeMap<String, Vec<UiShape>>,
    revision: u64,
}

impl UiState {
    /// The value held under `name`.
    pub fn get(&self, name: &str) -> Option<&UiValue> {
        self.values.get(name)
    }

    /// Hold `value` under `name`.
    pub fn set(&mut self, name: &str, value: UiValue) {
        if self.values.get(name) != Some(&value) {
            self.values.insert(name.to_string(), value);
            self.bump();
        }
    }

    /// Hold `value` under `name` unless something already is.
    pub fn seed(&mut self, name: &str, value: UiValue) {
        if !self.values.contains_key(name) {
            self.set(name, value);
        }
    }

    /// The items of list `name`, empty when the app has filled none.
    pub fn list(&self, name: &str) -> &[UiItem] {
        self.lists.get(name).map_or(&[], Vec::as_slice)
    }

    /// Replace list `name` with `items`.
    pub fn set_list(&mut self, name: &str, items: Vec<UiItem>) {
        if self.lists.get(name).map(Vec::as_slice) != Some(items.as_slice()) {
            self.lists.insert(name.to_string(), items);
            self.bump();
        }
    }

    /// Change list `name` in place, creating it empty when absent. Counts as a
    /// change whatever `edit` does.
    pub fn edit_list(&mut self, name: &str, edit: impl FnOnce(&mut Vec<UiItem>)) {
        if !self.lists.contains_key(name) {
            self.lists.insert(name.to_string(), Vec::new());
        }
        if let Some(items) = self.lists.get_mut(name) {
            edit(items);
        }
        self.bump();
    }

    /// The shapes of shape list `name`, empty when the app has filled none.
    pub fn shapes(&self, name: &str) -> &[UiShape] {
        self.shapes.get(name).map_or(&[], Vec::as_slice)
    }

    /// Change shape list `name` in place, creating it empty when absent, so a
    /// frame that only moves shapes allocates nothing. Shapes are read whole
    /// every frame, so a change does not count toward the revision.
    pub fn edit_shapes(&mut self, name: &str, edit: impl FnOnce(&mut Vec<UiShape>)) {
        if !self.shapes.contains_key(name) {
            self.shapes.insert(name.to_string(), Vec::new());
        }
        if let Some(shapes) = self.shapes.get_mut(name) {
            edit(shapes);
        }
    }

    /// A count that changes whenever a value or list does, so a reader can
    /// skip rereading an unchanged state.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn bump(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn an_image_fits_its_area_at_its_aspect() {
        let image = UiImage {
            texture: TextureHandle::new(0),
            size: [200, 100],
        };
        assert_eq!(
            image.fitted([0.0, 0.0, 80.0, 80.0]),
            [0.0, 20.0, 80.0, 40.0]
        );
        let blank = UiImage {
            size: [0, 0],
            ..image
        };
        assert_eq!(blank.fitted([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn state_counts_only_real_changes() {
        let mut state = UiState::default();
        state.set("a", UiValue::Bool(true));
        let after_set = state.revision();
        state.set("a", UiValue::Bool(true));
        assert_eq!(state.revision(), after_set, "an unchanged set");
        state.seed("a", UiValue::Bool(false));
        assert_eq!(
            state.get("a"),
            Some(&UiValue::Bool(true)),
            "seed keeps a held value"
        );
        state.seed("b", UiValue::Int(1));
        assert_eq!(state.get("b"), Some(&UiValue::Int(1)));
    }

    #[test]
    fn shapes_are_edited_in_place_without_a_revision() {
        let mut state = UiState::default();
        assert!(state.shapes("chart").is_empty());
        let before = state.revision();
        state.edit_shapes("chart", |s| {
            s.push(UiShape {
                tag: Some(3),
                ..UiShape::default()
            });
        });
        assert_eq!(state.shapes("chart")[0].tag, Some(3));
        assert_eq!(state.revision(), before);
    }

    #[test]
    fn lists_are_filled_whole_or_edited_in_place() {
        let mut state = UiState::default();
        assert!(state.list("files").is_empty());
        state.set_list("files", vec![UiItem::new("a"), UiItem::new("b")]);
        let filled = state.revision();
        state.set_list("files", vec![UiItem::new("a"), UiItem::new("b")]);
        assert_eq!(state.revision(), filled, "the same items");
        state.edit_list("files", |items| items[1].expanded = true);
        assert_ne!(state.revision(), filled);
        assert!(state.list("files")[1].expanded);
        state.edit_list("new", |items| items.push(UiItem::new("c")));
        assert_eq!(state.list("new").len(), 1);
    }
}

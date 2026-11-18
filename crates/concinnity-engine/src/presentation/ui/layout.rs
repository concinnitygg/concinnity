// Layout: rows, columns, and stacks of overlay elements, sized fixed, to their
// content, or to a share of the space left, and solved each frame.

use crate::components::{Screen, TextLabel, Vocabulary};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Ref, RefTarget};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

/// How a [Layout](#layout) arranges its children.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum LayoutDirection {
    /// Top to bottom.
    #[default]
    #[vocab("column")]
    Column,
    /// Left to right.
    #[vocab("row")]
    Row,
    /// On top of one another, each placed in the whole layout.
    #[vocab("stack")]
    Stack,
}

/// Where a [Layout](#layout)'s children sit along an axis when they leave space
/// over.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutAlign {
    /// Against the start: the left or the top.
    #[default]
    #[vocab("start")]
    Start,
    /// Centered.
    #[vocab("center")]
    Center,
    /// Against the end: the right or the bottom.
    #[vocab("end")]
    End,
    /// Along the main axis, spread out with the space left shared between the
    /// children; across it, the same as `start`.
    #[vocab("space-between")]
    SpaceBetween,
}

/// Where a root [Layout](#layout) sits on its screen.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutAnchor {
    /// The top-left corner.
    #[default]
    #[vocab("top-left")]
    TopLeft,
    /// The middle of the top edge.
    #[vocab("top")]
    Top,
    /// The top-right corner.
    #[vocab("top-right")]
    TopRight,
    /// The middle of the left edge.
    #[vocab("left")]
    Left,
    /// The center.
    #[vocab("center")]
    Center,
    /// The middle of the right edge.
    #[vocab("right")]
    Right,
    /// The bottom-left corner.
    #[vocab("bottom-left")]
    BottomLeft,
    /// The middle of the bottom edge.
    #[vocab("bottom")]
    Bottom,
    /// The bottom-right corner.
    #[vocab("bottom-right")]
    BottomRight,
}

impl LayoutAnchor {
    // How far across the free space the anchor sits, horizontally and
    // vertically: 0 at the start, 1 at the end.
    pub(crate) fn fractions(self) -> [f32; 2] {
        match self {
            Self::TopLeft => [0.0, 0.0],
            Self::Top => [0.5, 0.0],
            Self::TopRight => [1.0, 0.0],
            Self::Left => [0.0, 0.5],
            Self::Center => [0.5, 0.5],
            Self::Right => [1.0, 0.5],
            Self::BottomLeft => [0.0, 1.0],
            Self::Bottom => [0.5, 1.0],
            Self::BottomRight => [1.0, 1.0],
        }
    }
}

/// How one axis of a [Layout](#layout) child is sized: `"fit"`, `{"fixed": 120}`,
/// or `{"fill": 1}`.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayoutSize {
    /// To its content: a label to its measured text, an element to its own
    /// size, a layout to its children.
    #[default]
    Fit,
    /// This many pixels.
    Fixed(f32),
    /// A share of the space its siblings leave, by this weight.
    Fill(f32),
}

#[cfg(feature = "schema")]
impl crate::ecs::schema::Described for LayoutSize {
    const TYPE: crate::ecs::schema::FieldType = crate::ecs::schema::FieldType::Enum(&SIZE);
}

#[cfg(feature = "schema")]
static SIZE: crate::ecs::schema::TypeSchema = {
    use crate::ecs::schema::{Body, FieldType, TypeSchema, VariantSchema};
    TypeSchema {
        name: "LayoutSize",
        doc: "How one axis of a [Layout](#layout) child is sized.",
        body: Body::Variants(&[
            VariantSchema {
                name: "fit",
                doc: "To its content: a label to its measured text, an element to its own size, a layout to its children.",
                payload: None,
            },
            VariantSchema {
                name: "fixed",
                doc: "This many pixels.",
                payload: Some(|| FieldType::Float),
            },
            VariantSchema {
                name: "fill",
                doc: "A share of the space its siblings leave, by this weight.",
                payload: Some(|| FieldType::Float),
            },
        ]),
        default: None,
    }
};

/// What a [Layout](#layout) child names: a [Sprite](#sprite),
/// [`TextLabel`](#textlabel), [`TextField`](#textfield), a widget
/// ([List](#list), [Tabs](#tabs), [Checkbox](#checkbox), or
/// [`TextArea`](#textarea)), or another [Layout](#layout).
#[derive(Debug, Clone, Copy)]
pub struct LayoutItem;

impl RefTarget for LayoutItem {
    const TYPES: &'static [&'static str] = &[
        "Sprite",
        "TextLabel",
        "TextField",
        "List",
        "Tabs",
        "Checkbox",
        "TextArea",
        "Layout",
    ];
}

/// One child of a [Layout](#layout), and how it is sized.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct LayoutChild {
    /// The element or layout placed.
    pub item: Option<Ref<LayoutItem>>,
    /// How wide it is.
    pub width: LayoutSize,
    /// How tall it is.
    pub height: LayoutSize,
}

/// Places overlay elements in a row, a column, or a stack, so a screen does not
/// hand-position every piece.
///
/// Each child is sized per axis: to its content (`fit`), to a fixed size, or to
/// a share of the space its siblings leave (`fill`). A label fits its measured
/// text, so the layout reflows as live text changes width. A hidden child, or a
/// label with no text, takes no space, and a layout with nothing to show takes
/// none either. The layout owns its children's position, and the size of a
/// [Sprite](#sprite) or [`TextField`](#textfield) it sizes; a label keeps the
/// size of its text.
///
/// A layout no other layout holds is a root: `width` and `height` size it, and
/// `anchor` and `margin` place it on its [Screen](#screen), the reference
/// canvas or the window by the screen's `space`.
///
/// A `movable` root is a floating panel: a press on the band `grab` tall
/// across its top drags it about its screen. A `resizable` one is sized by
/// dragging its right or bottom edge or its corner, never smaller than
/// `min_size`.
///
/// ```rust
/// # use concinnity_engine::components::{Layout, LayoutAnchor, LayoutDirection};
/// Layout {
///     direction: LayoutDirection::Row,
///     anchor: LayoutAnchor::TopRight,
///     margin: [10.0, 10.0],
///     gap: 6.0,
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
pub struct Layout {
    /// [Screen](#screen) the layout belongs to, whose space a root is placed
    /// in.
    pub screen: Option<Ref<Screen>>,
    /// How the children are arranged.
    pub direction: LayoutDirection,
    /// The children, in order.
    pub children: Vec<LayoutChild>,
    /// Pixels between adjacent children.
    pub gap: f32,
    /// Pixels kept clear inside every edge.
    pub padding: f32,
    /// Where the children sit along the direction when they leave space over
    /// (horizontally in a stack).
    pub justify: LayoutAlign,
    /// Where each child sits across the direction (vertically in a stack).
    pub align: LayoutAlign,
    /// A root's width.
    pub width: LayoutSize,
    /// A root's height.
    pub height: LayoutSize,
    /// The point of its screen a root sits at.
    pub anchor: LayoutAnchor,
    /// A root's distance from the screen edges it is anchored to, `[x, y]`.
    pub margin: [f32; 2],
    /// When false, the layout leaves its children where they are and takes no
    /// space in its parent.
    #[asset(default = true)]
    pub visible: bool,
    /// Whether a root is dragged about its screen by the band across its top.
    pub movable: bool,
    /// Whether a root is sized by dragging its right or bottom edge.
    pub resizable: bool,
    /// Height of the band across a movable root's top that drags it.
    #[asset(default = 28.0)]
    pub grab: f32,
    /// The smallest a resizable root is dragged to, `[width, height]`.
    #[asset(default = [120.0, 80.0])]
    pub min_size: [f32; 2],
    /// Runtime: how far a drag has moved the root from where it is anchored,
    /// `[x, y]`. Not authored.
    #[serde(skip)]
    pub offset: [f32; 2],
    /// Runtime: the size a drag has given the root, in place of `width` and
    /// `height`. Not authored.
    #[serde(skip)]
    pub size: Option<[f32; 2]>,
    /// Runtime: the root's rectangle as last laid out. Not authored.
    #[serde(skip)]
    pub rect: Option<[f32; 4]>,
}

/// A label's measured extent: the background-box size in pixels, and the inset
/// from the box's top-left corner to the text origin, which a layout places the
/// box by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelBox {
    /// Background-box width in pixels.
    pub w: f32,
    /// Background-box height in pixels.
    pub h: f32,
    /// Horizontal inset from the box's left edge to the text origin.
    pub pad: f32,
    /// Vertical inset from the box's top edge down to the text origin.
    pub top_inset: f32,
}

// What the solver reads: the layouts by id, and each element's natural size,
// `None` for one that takes no space.
pub(crate) trait LayoutSource {
    fn layout(&self, id: AssetId) -> Option<&Layout>;
    fn element(&self, id: AssetId) -> Option<[f32; 2]>;
}

// Deeper nesting than this is taken for a reference cycle and placed no further.
const MAX_DEPTH: usize = 32;

// Reused per frame so a solve allocates nothing in steady state.
#[derive(Debug, Default)]
pub(crate) struct LayoutSolve {
    // Each nested layout's content size this solve, `None` when it shows
    // nothing or is still being sized (a layout that holds itself).
    natural: HashMap<AssetId, Option<[f32; 2]>>,
    // The layouts being placed right now, so one holding itself is placed
    // once.
    placing: HashSet<AssetId>,
    // Every element placed, as `[x, y, width, height]`.
    pub(crate) placed: Vec<(AssetId, [f32; 4])>,
}

impl LayoutSolve {
    pub(crate) fn clear(&mut self) {
        self.natural.clear();
        self.placing.clear();
        self.placed.clear();
    }

    // Place `root`, named `id`, and everything it holds within a screen of
    // `area`, adding each element's rect to `placed`. Returns the root's rect,
    // or `None` when it shows nothing.
    pub(crate) fn solve_root<S: LayoutSource>(
        &mut self,
        src: &S,
        id: Option<AssetId>,
        root: &Layout,
        area: [f32; 2],
    ) -> Option<[f32; 4]> {
        if let Some(id) = id {
            self.natural.insert(id, None);
            self.placing.insert(id);
        }
        let content = self.content(src, root, 0);
        if let Some(id) = id {
            self.natural.insert(id, content);
        }
        let content = content?;
        let size = |axis: usize| match (root.size, root.size(axis)) {
            (Some(dragged), _) => dragged[axis],
            (None, LayoutSize::Fixed(v)) => v,
            (None, LayoutSize::Fit) => content[axis],
            (None, LayoutSize::Fill(_)) => (area[axis] - 2.0 * root.margin[axis]).max(0.0),
        };
        let (w, h) = (size(0), size(1));
        let [fx, fy] = root.anchor.fractions();
        let place = |axis: usize, len: f32, frac: f32| {
            let free = area[axis] - 2.0 * root.margin[axis] - len;
            root.margin[axis] + frac * free + root.offset[axis]
        };
        let rect = [place(0, w, fx), place(1, h, fy), w, h];
        self.place(src, root, rect, 0);
        if let Some(id) = id {
            self.placing.remove(&id);
        }
        Some(rect)
    }

    // A child's natural size: an element's own, a layout's content.
    fn natural<S: LayoutSource>(&mut self, src: &S, id: AssetId, depth: usize) -> Option<[f32; 2]> {
        if let Some(layout) = src.layout(id) {
            if let Some(&known) = self.natural.get(&id) {
                return known;
            }
            // Marked before sizing, so a layout reached again inside itself
            // reads as taking no space.
            self.natural.insert(id, None);
            let size = self.content(src, layout, depth + 1);
            self.natural.insert(id, size);
            return size;
        }
        src.element(id)
    }

    // A child's size before any fill share: fixed, its content, or nothing
    // along a fill axis.
    fn base<S: LayoutSource>(
        &mut self,
        src: &S,
        child: &LayoutChild,
        depth: usize,
    ) -> Option<[f32; 2]> {
        let id = child.item?.id();
        // A layout already being placed is this one's own ancestor: holding it
        // again takes no space.
        if self.placing.contains(&id) {
            return None;
        }
        let natural = self.natural(src, id, depth)?;
        let axis = |size: LayoutSize, n: f32| match size {
            LayoutSize::Fixed(v) => v,
            LayoutSize::Fit => n,
            LayoutSize::Fill(_) => 0.0,
        };
        Some([
            axis(child.width, natural[0]),
            axis(child.height, natural[1]),
        ])
    }

    // What `layout`'s children need, padding included, or `None` when none of
    // them shows.
    fn content<S: LayoutSource>(
        &mut self,
        src: &S,
        layout: &Layout,
        depth: usize,
    ) -> Option<[f32; 2]> {
        if !layout.visible || depth > MAX_DEPTH {
            return None;
        }
        let main = layout.direction.main();
        let mut size = [0.0_f32; 2];
        let mut shown = 0usize;
        for child in &layout.children {
            let Some(base) = self.base(src, child, depth) else {
                continue;
            };
            shown += 1;
            match main {
                Some(m) => {
                    size[m] += base[m];
                    size[1 - m] = size[1 - m].max(base[1 - m]);
                }
                None => size = [size[0].max(base[0]), size[1].max(base[1])],
            }
        }
        if shown == 0 {
            return None;
        }
        if let Some(m) = main {
            size[m] += layout.gap * (shown - 1) as f32;
        }
        Some([
            size[0] + 2.0 * layout.padding,
            size[1] + 2.0 * layout.padding,
        ])
    }

    // Place `layout`'s children within `rect`.
    fn place<S: LayoutSource>(&mut self, src: &S, layout: &Layout, rect: [f32; 4], depth: usize) {
        if depth > MAX_DEPTH {
            return;
        }
        let pad = layout.padding;
        let inner = [
            rect[0] + pad,
            rect[1] + pad,
            (rect[2] - 2.0 * pad).max(0.0),
            (rect[3] - 2.0 * pad).max(0.0),
        ];
        let Some(m) = layout.direction.main() else {
            for child in &layout.children {
                if let Some(base) = self.base(src, child, depth) {
                    let along = |axis: usize, align: LayoutAlign| {
                        let len = match child.size(axis) {
                            LayoutSize::Fill(_) => inner[2 + axis],
                            _ => base[axis],
                        };
                        let at = inner[axis] + align.offset(inner[2 + axis] - len);
                        (at, len)
                    };
                    let (x, w) = along(0, layout.justify);
                    let (y, h) = along(1, layout.align);
                    self.put(src, child, [x, y, w, h], depth);
                }
            }
            return;
        };

        // Along the direction: the fixed and fitted sizes and the gaps come
        // first, and the fills share what is left by weight.
        let (mut used, mut weight, mut shown) = (0.0_f32, 0.0_f32, 0usize);
        for child in &layout.children {
            if let Some(base) = self.base(src, child, depth) {
                shown += 1;
                used += base[m];
                if let LayoutSize::Fill(w) = child.size(m) {
                    weight += w.max(0.0);
                }
            }
        }
        if shown == 0 {
            return;
        }
        let gaps = layout.gap * (shown - 1) as f32;
        let free = (inner[2 + m] - used - gaps).max(0.0);
        let (mut at, gap) = if weight > 0.0 {
            (0.0, layout.gap)
        } else if layout.justify == LayoutAlign::SpaceBetween && shown > 1 {
            (0.0, layout.gap + free / (shown - 1) as f32)
        } else {
            (layout.justify.offset(free), layout.gap)
        };
        for child in &layout.children {
            let Some(base) = self.base(src, child, depth) else {
                continue;
            };
            let along = match child.size(m) {
                LayoutSize::Fill(w) if weight > 0.0 => free * w.max(0.0) / weight,
                _ => base[m],
            };
            let across = match child.size(1 - m) {
                LayoutSize::Fill(_) => inner[2 + (1 - m)],
                _ => base[1 - m],
            };
            let cross_at = inner[1 - m] + layout.align.offset(inner[2 + (1 - m)] - across);
            let mut r = [0.0; 4];
            r[m] = inner[m] + at;
            r[1 - m] = cross_at;
            r[2 + m] = along;
            r[2 + (1 - m)] = across;
            self.put(src, child, r, depth);
            at += along + gap;
        }
    }

    // A placed child: a layout lays out its own children within `rect`, an
    // element takes it.
    fn put<S: LayoutSource>(&mut self, src: &S, child: &LayoutChild, rect: [f32; 4], depth: usize) {
        let Some(id) = child.item.map(Ref::id) else {
            return;
        };
        match src.layout(id) {
            Some(nested) => {
                if self.placing.insert(id) {
                    self.place(src, nested, rect, depth + 1);
                    self.placing.remove(&id);
                }
            }
            None => self.placed.push((id, rect)),
        }
    }
}

impl LayoutDirection {
    // The axis children follow, 0 for x and 1 for y; `None` for a stack.
    fn main(self) -> Option<usize> {
        match self {
            Self::Row => Some(0),
            Self::Column => Some(1),
            Self::Stack => None,
        }
    }
}

impl LayoutAlign {
    // How far into `free` space a child sits.
    fn offset(self, free: f32) -> f32 {
        match self {
            Self::Start | Self::SpaceBetween => 0.0,
            Self::Center => free.max(0.0) / 2.0,
            Self::End => free.max(0.0),
        }
    }
}

impl LayoutChild {
    // A child of `item` sized as it fits on both axes.
    pub(crate) fn fit(item: AssetId) -> Self {
        Self {
            item: Some(Ref::new(item)),
            ..Self::default()
        }
    }

    fn size(&self, axis: usize) -> LayoutSize {
        if axis == 0 { self.width } else { self.height }
    }
}

impl Layout {
    fn size(&self, axis: usize) -> LayoutSize {
        if axis == 0 { self.width } else { self.height }
    }
}

// Whether a label takes space in a layout: a hidden or empty one does not.
pub(crate) fn label_present(label: &TextLabel) -> bool {
    label.visible && !label.content.is_empty()
}

#[cfg(test)]
mod tests;

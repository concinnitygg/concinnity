// Layout: rows, columns, and stacks of overlay elements, sized fixed, to their
// content, or to a share of the space left, and solved each frame.

use crate::components::{Screen, TextLabel, Vocabulary};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Ref, RefTarget};
use crate::presentation::ui::widgets::scroll_view::ScrollView;
use alloc::string::String;
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
/// ([List](#list), [Grid](#grid), [Canvas](#canvas), [`ValueSlider`](#valueslider),
/// [Tabs](#tabs), [Checkbox](#checkbox), [Button](#button), or
/// [`TextArea`](#textarea)), or another [Layout](#layout).
#[derive(Debug, Clone, Copy)]
pub struct LayoutItem;

impl RefTarget for LayoutItem {
    const TYPES: &'static [&'static str] = &[
        "Sprite",
        "TextLabel",
        "TextField",
        "List",
        "Grid",
        "Canvas",
        "ValueSlider",
        "Tabs",
        "Checkbox",
        "Button",
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
/// `min_size`. Floating panels overlap: a press on one brings it in front of
/// the others, and the pointer reaches only the frontmost panel under it.
///
/// A `front` root stays in front of the floating panels on its screen, such
/// as a toolbar: a press never brings a panel over it, and a panel under it is
/// covered where it lies.
///
/// A layout that names an `on_dismiss` event is a transient one, such as a
/// palette or a picker: while it shows, a press outside it sends the event, so
/// the app can close it. A press outside every such layout goes no further,
/// so it never acts on what lies under them; one inside another of them stays
/// that layout's own.
///
/// A `scroll` column holds children taller than it: they scroll with the wheel
/// and a scrollbar along its right edge, and draw and take input only inside
/// it. It asks its parent for no height of its own, so it is sized to fill or
/// to a fixed height.
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
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent flags of one layout, each read on its own"
)]
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
    /// When false, the layout and everything in it is hidden: it takes no space
    /// in its parent, and its elements neither draw nor take input.
    #[asset(default = true)]
    pub visible: bool,
    /// Whether a root is dragged about its screen by the band across its top.
    pub movable: bool,
    /// Whether a root is sized by dragging its right or bottom edge.
    pub resizable: bool,
    /// Whether a column scrolls children taller than itself, clipped to it.
    pub scroll: bool,
    /// Whether a root stays in front of the floating panels on its screen.
    pub front: bool,
    /// Whether a root shown with it set takes the input of its screen and the
    /// screens layered beneath it, as a dialog does: it stacks in front of the
    /// other roots on its screen's layer and dims what is behind it, and only
    /// what it holds takes the pointer, a menu or toast click, or keeps the
    /// keyboard. Screens layered above it are untouched.
    pub modal: bool,
    /// The event a press outside the shown layout sends. A press outside every
    /// layout naming one goes no further.
    pub on_dismiss: String,
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
    /// Runtime: the rectangle of a root, a scrolling layout, or one with an
    /// `on_dismiss` event as last laid out; `None` while it is not laid out.
    /// Not authored.
    #[serde(skip)]
    pub rect: Option<[f32; 4]>,
    /// Runtime: how far a scrolling layout's children are scrolled up, in
    /// pixels. Not authored.
    #[serde(skip)]
    pub scrolled: f32,
    /// Runtime: the height of a scrolling layout's children as last laid out,
    /// padding included. Not authored.
    #[serde(skip)]
    pub content_height: f32,
    /// Runtime: where a movable or resizable root stacks among the others on
    /// its screen, higher in front; a press on one brings it to the front.
    /// Not authored.
    #[serde(skip)]
    pub rank: u32,
    /// Runtime: the edges a drag is sizing the root by, while one is. Not
    /// authored.
    #[serde(skip)]
    pub resizing: Option<ResizeEdges>,
}

/// The edges of a resizable root a drag sizes it by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeEdges {
    /// The right edge, which sizes its width.
    pub right: bool,
    /// The bottom edge, which sizes its height.
    pub bottom: bool,
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
    // Every scrolling layout placed: its rect and its children's height.
    pub(crate) scrolls: Vec<(AssetId, [f32; 4], f32)>,
    // The nested layouts with an `on_dismiss` event, where they were placed.
    pub(crate) dismissible: Vec<(AssetId, [f32; 4])>,
}

impl LayoutSolve {
    pub(crate) fn clear(&mut self) {
        self.natural.clear();
        self.placing.clear();
        self.placed.clear();
        self.scrolls.clear();
        self.dismissible.clear();
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
            let size = self.content(src, layout, depth + 1).map(|[w, h]| {
                // A scrolling layout takes the height it is given, not its
                // children's.
                let h = if layout.scroll {
                    2.0 * layout.padding
                } else {
                    h
                };
                [w, h]
            });
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
                if !nested.on_dismiss.is_empty() {
                    self.dismissible.push((id, rect));
                }
                if self.placing.insert(id) {
                    if nested.scroll {
                        self.place_scrolled(src, id, nested, rect, depth + 1);
                    } else {
                        self.place(src, nested, rect, depth + 1);
                    }
                    self.placing.remove(&id);
                }
            }
            None => self.placed.push((id, rect)),
        }
    }

    // Place a scrolling layout's children at their full height, moved up by
    // its scroll and narrowed by its scrollbar while they overflow `rect`.
    fn place_scrolled<S: LayoutSource>(
        &mut self,
        src: &S,
        id: AssetId,
        layout: &Layout,
        rect: [f32; 4],
        depth: usize,
    ) {
        let content = self.content(src, layout, depth).map_or(0.0, |c| c[1]);
        let view = ScrollView {
            rect,
            content,
            scroll: layout.scrolled,
        };
        let scroll = layout.scrolled.clamp(0.0, view.max_scroll());
        let inner = [
            rect[0],
            rect[1] - scroll,
            view.inner_width(),
            rect[3].max(content),
        ];
        self.place(src, layout, inner, depth);
        self.scrolls.push((id, rect, content));
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
    /// How far inside a resizable root's right and bottom edges a press takes
    /// the edge.
    pub const RESIZE_EDGE: f32 = 6.0;

    fn size(&self, axis: usize) -> LayoutSize {
        if axis == 0 { self.width } else { self.height }
    }

    /// The edges a press at `(x, y)` would size this root by: those of a shown,
    /// resizable root it lands within [`RESIZE_EDGE`](Self::RESIZE_EDGE) of,
    /// where the root was last laid out.
    pub fn resize_grip(&self, x: f32, y: f32) -> Option<ResizeEdges> {
        let rect = self.rect.filter(|_| self.resizable && self.visible)?;
        if !super::widgets::point_in(x, y, rect) {
            return None;
        }
        let edges = ResizeEdges {
            right: x >= rect[0] + rect[2] - Self::RESIZE_EDGE,
            bottom: y >= rect[1] + rect[3] - Self::RESIZE_EDGE,
        };
        (edges.right || edges.bottom).then_some(edges)
    }
}

// Whether a label takes space in a layout: a hidden or empty one does not.
pub(crate) fn label_present(label: &TextLabel) -> bool {
    label.visible && !label.content.is_empty()
}

#[cfg(test)]
mod tests;

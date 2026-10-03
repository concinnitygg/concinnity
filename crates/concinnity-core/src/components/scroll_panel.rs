// Scrollable UI panel schema.

use crate::components::{Screen, Sprite, TextLabel};
use crate::ecs::{Ref, RefTarget};
use alloc::string::String;
use alloc::vec::Vec;

/// Runtime model that makes a band of UI rows scrollable and (optionally)
/// collapsible.
///
/// A `ScrollPanel` is emitted by the build (e.g. by a settings menu) and read
/// by the UI at runtime; it is not hand-authored. It names a content band (a
/// fixed rectangle in the menu's reference canvas), the ordered rows that live
/// inside it, the collapsible groups some rows belong to, and the scrollbar
/// thumb/track sprites. The UI lays the rows out each frame: a collapsed group's
/// body rows hide and the rows below them move up; when the visible stack is
/// taller than the band it scrolls (mouse wheel or thumb drag) and rows outside
/// the band are clipped.
///
/// All pixel fields are in the same reference-space coordinates as the Screen's
/// other UI (see the overlay scaling notes on [MainMenu](#mainmenu)).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct ScrollPanel {
    /// [Screen](#screen) this panel belongs to. The panel is only live while
    /// its screen is active.
    pub screen: Option<Ref<Screen>>,
    /// Left edge of the content band in reference pixels.
    pub x: f32,
    /// Top edge of the content band in reference pixels.
    pub y: f32,
    /// Width of the content band in reference pixels.
    pub width: f32,
    /// Height of the content band (the visible window) in reference pixels.
    pub height: f32,
    /// The rows in the band, top to bottom.
    pub rows: Vec<ScrollRow>,
    /// Collapsible groups, referenced by index from [ScrollRow::group].
    pub groups: Vec<ScrollGroup>,
    /// Scrollbar thumb [Sprite](#sprite) the UI moves and resizes. `None` for a
    /// panel with no scrollbar.
    pub thumb: Option<Ref<Sprite>>,
    /// Scrollbar track [Sprite](#sprite). Hidden along with the thumb when the
    /// content fits the band.
    pub track: Option<Ref<Sprite>>,
    /// Left edge of the scrollbar track in reference pixels.
    pub track_x: f32,
    /// Top edge of the scrollbar track in reference pixels.
    pub track_y: f32,
    /// Width of the scrollbar track in reference pixels.
    pub track_w: f32,
    /// Height of the scrollbar track in reference pixels (the thumb travels
    /// within it).
    pub track_h: f32,
}

/// One row inside a [ScrollPanel](#scrollpanel): the elements that move
/// together, the row's height, and the collapsible group it belongs to.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ScrollRow {
    /// The [Sprite](#sprite)/[TextLabel](#textlabel) ids that make up this row
    /// and move (and clip) together. Click regions are matched to their row by
    /// position, so they are not listed here.
    pub elements: Vec<Ref<ScrollElement>>,
    /// The row's authored top edge in reference pixels (its build-time, all
    /// groups expanded, unscrolled position).
    pub base_y: f32,
    /// The row's height in reference pixels (its vertical pitch in the stack).
    pub height: f32,
    /// Index into [ScrollPanel::groups] of the group whose collapsed state
    /// hides this row, or `-1` for a row that is always shown (a group header
    /// or an ungrouped row).
    #[asset(default = -1)]
    pub group: i32,
}

/// What a [ScrollRow](#scrollrow) element may name: a [Sprite](#sprite) or a
/// [TextLabel](#textlabel).
#[derive(Debug, Clone, Copy)]
pub struct ScrollElement;

impl RefTarget for ScrollElement {
    const TYPES: &'static [&'static str] = &["Sprite", "TextLabel"];
}

/// A collapsible group of rows inside a [ScrollPanel](#scrollpanel).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct ScrollGroup {
    /// Whether the group starts collapsed (its body rows hidden).
    pub collapsed: bool,
    /// The header [TextLabel](#textlabel) whose text gets a `+`/`-` prefix to
    /// reflect the collapsed state. `None` leaves the header text unchanged.
    pub header: Option<Ref<TextLabel>>,
    /// The header's base title (e.g. `"Advanced"`); the UI shows `"+ Advanced"`
    /// when collapsed and `"- Advanced"` when expanded.
    pub title: String,
}

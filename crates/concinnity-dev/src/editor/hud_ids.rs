//! The editor HUD's reserved asset ids. Every module that injects HUD elements
//! declares them once with `hud_ids!`, in draw order, under a base taken from
//! its family here. Families sit a fixed stride apart, so no two modules can
//! hand out the same id, and the macro fails the build for a family that
//! outgrows its stride.
//!
//! The ids are never serialized: they exist only in a live editor session, so
//! their numeric values carry no meaning beyond being unique.

use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::panels::registry::PanelKey;
use super::widget;

// Base of the reserved range. Interned world ids are dense from 0 and never
// approach it.
pub(crate) const ID_BASE: u32 = 0x3000_0000;

// The ids one family may take.
pub(crate) const FAMILY_SPAN: usize = 0x1000;

// The HUD id families outside the floating panels. Each panel is a family of its
// own, numbered after these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    TopBar,
    Highlight,
    Gizmo,
    Marquee,
    Cursor,
    Billboards,
    CreateMenu,
    DisplayMenu,
    Toasts,
    Modal,
    ShotFade,
    Loading,
}

const FAMILY_COUNT: u32 = Family::Loading as u32 + 1;

pub(crate) const fn family_base(family: Family) -> u32 {
    nth_base(family as u32)
}

pub(crate) const fn panel_base(key: PanelKey) -> u32 {
    nth_base(FAMILY_COUNT + key as u32)
}

const fn nth_base(n: u32) -> u32 {
    ID_BASE + n * FAMILY_SPAN as u32
}

// One module's injected elements, each list in draw order (the overlay draws
// in insertion order, not by id). Fields carry their placeholder text.
#[derive(Debug, Default)]
pub(crate) struct HudIds {
    pub(crate) sprites: Vec<AssetId>,
    pub(crate) labels: Vec<AssetId>,
    // Labels drawn in the monospace code face rather than the HUD face.
    pub(crate) code_labels: Vec<AssetId>,
    pub(crate) fields: Vec<(AssetId, &'static str)>,
}

impl HudIds {
    pub(crate) fn field_ids(&self) -> impl Iterator<Item = AssetId> + '_ {
        self.fields.iter().map(|&(id, _)| id)
    }

    // Every id, sprites first.
    pub(crate) fn all(&self) -> impl Iterator<Item = AssetId> + '_ {
        self.sprites
            .iter()
            .chain(&self.labels)
            .chain(&self.code_labels)
            .copied()
            .chain(self.field_ids())
    }

    // Blank every element. Fields are blurred as well, so a hidden field cannot
    // keep keyboard focus.
    pub(crate) fn hide(&self, world: &mut World) {
        for &id in &self.sprites {
            widget::set_sprite_visible(world, id, false);
        }
        for &id in self.labels.iter().chain(&self.code_labels) {
            widget::set_label_visible(world, id, false);
        }
        for id in self.field_ids() {
            widget::hide_field(world, id);
        }
    }
}

#[cfg(test)]
impl HudIds {
    // A world holding one blank element per id (code labels as labels).
    pub(crate) fn test_world(&self) -> World {
        let labels: Vec<AssetId> = self
            .labels
            .iter()
            .chain(&self.code_labels)
            .copied()
            .collect();
        let fields: Vec<AssetId> = self.field_ids().collect();
        crate::test_support::injected_world(&self.sprites, &labels, &fields)
    }
}

// Declare a module's HUD ids under `base`, one list per element kind
// (`sprites`, `labels`, `code_labels`, `fields`), each in draw order:
//
//   hud_ids! {
//       base: panel_base(PanelKey::Health);
//       sprites: [pub(crate) PANEL_BG, track[ROWS], [ROWS] { used, fill }, ..AREA.sprite_ids()];
//       fields: [INPUT = "search"];
//       blocks: [AREA_BASE: TextAreaIds::SPAN];
//   }
//
// `NAME` is one id, `name[N]` a pool of `N` read through `name(i)`, and
// `[N] { a, b }` pools drawn interleaved (`a(0)`, `b(0)`, `a(1)`, ...). A field
// takes its placeholder after `=`. `..ids` splices in ids declared elsewhere
// (a widget drawn inside the panel), whose room is a `blocks` entry: a `u32`
// base with that many ids reserved behind it. Ids are numbered in declaration
// order from `base`. The macro also defines `ids()`, the module's `HudIds`, and
// `IDS_SPAN`, how many ids it took.
macro_rules! hud_ids {
    (base: $base:expr; $($kind:ident: [$($entry:tt)*];)*) => {
        const IDS_BASE: u32 = $base;
        $crate::editor::hud_ids::hud_ids!(@next ids [0usize] [] $($kind [$($entry)*])*);
    };

    (@next $ids:ident [$($off:tt)*] [$($acc:tt)*]) => {
        pub(crate) const IDS_SPAN: usize = $($off)*;
        const _: () = assert!(IDS_SPAN <= $crate::editor::hud_ids::FAMILY_SPAN);

        pub(crate) fn ids() -> &'static $crate::editor::hud_ids::HudIds {
            static IDS: std::sync::LazyLock<$crate::editor::hud_ids::HudIds> =
                std::sync::LazyLock::new(|| {
                    let mut $ids = $crate::editor::hud_ids::HudIds::default();
                    $($acc)*
                    $ids
                });
            &IDS
        }
    };
    (@next $ids:ident $off:tt $acc:tt $kind:ident [$($entry:tt)*] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind $off $acc [$($entry)*] $($rest)*);
    };

    (@entry $ids:ident $kind:ident $off:tt $acc:tt [] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@next $ids $off $acc $($rest)*);
    };

    // `blocks: [NAME: span]`: a base with `span` ids reserved behind it.
    (@entry $ids:ident blocks [$($off:tt)*] $acc:tt
        [$vis:vis $name:ident : $span:expr $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $vis const $name: u32 = IDS_BASE + ($($off)*) as u32;
        $crate::editor::hud_ids::hud_ids!(@entry $ids blocks [$($off)* + ($span)] $acc
            [$($($tail)*)?] $($rest)*);
    };

    // `fields: [name[N] = "placeholder"]` and `fields: [NAME = "placeholder"]`.
    (@entry $ids:ident fields [$($off:tt)*] [$($acc:tt)*]
        [$vis:vis $name:ident [$n:expr] = $ph:expr $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@pool $vis $name [$($off)*]);
        $crate::editor::hud_ids::hud_ids!(@entry $ids fields [$($off)* + ($n)]
            [$($acc)* $ids.fields.extend((0..$n).map(|i| ($name(i), $ph)));]
            [$($($tail)*)?] $($rest)*);
    };
    (@entry $ids:ident fields [$($off:tt)*] [$($acc:tt)*]
        [$vis:vis $name:ident = $ph:expr $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $vis const $name: ::concinnity_core::ecs::asset_id::AssetId =
            ::concinnity_core::ecs::asset_id::AssetId(IDS_BASE + ($($off)*) as u32);
        $crate::editor::hud_ids::hud_ids!(@entry $ids fields [$($off)* + 1]
            [$($acc)* $ids.fields.push(($name, $ph));]
            [$($($tail)*)?] $($rest)*);
    };

    // `..ids`: ids declared elsewhere, drawn here.
    (@entry $ids:ident $kind:ident $off:tt [$($acc:tt)*]
        [.. $splice:expr $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind $off
            [$($acc)* $ids.$kind.extend($splice);]
            [$($($tail)*)?] $($rest)*);
    };

    // `[N] { a, b }`: one pool per member, drawn interleaved. The members are
    // numbered as plain pools (`@slot`) once the draw order is recorded.
    (@entry $ids:ident $kind:ident $off:tt [$($acc:tt)*]
        [[$n:expr] { $($vis:vis $name:ident),+ $(,)? } $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind $off
            [$($acc)* for i in 0..$n { $($ids.$kind.push($name(i));)+ }]
            [$(@slot $vis $name [$n],)+ $($($tail)*)?] $($rest)*);
    };
    (@entry $ids:ident $kind:ident [$($off:tt)*] $acc:tt
        [@slot $vis:vis $name:ident [$n:expr], $($tail:tt)*] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@pool $vis $name [$($off)*]);
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind [$($off)* + ($n)] $acc
            [$($tail)*] $($rest)*);
    };

    // `name[N]`.
    (@entry $ids:ident $kind:ident [$($off:tt)*] [$($acc:tt)*]
        [$vis:vis $name:ident [$n:expr] $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $crate::editor::hud_ids::hud_ids!(@pool $vis $name [$($off)*]);
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind [$($off)* + ($n)]
            [$($acc)* $ids.$kind.extend((0..$n).map($name));]
            [$($($tail)*)?] $($rest)*);
    };

    // `NAME`.
    (@entry $ids:ident $kind:ident [$($off:tt)*] [$($acc:tt)*]
        [$vis:vis $name:ident $(, $($tail:tt)*)?] $($rest:tt)*) => {
        $vis const $name: ::concinnity_core::ecs::asset_id::AssetId =
            ::concinnity_core::ecs::asset_id::AssetId(IDS_BASE + ($($off)*) as u32);
        $crate::editor::hud_ids::hud_ids!(@entry $ids $kind [$($off)* + 1]
            [$($acc)* $ids.$kind.push($name);]
            [$($($tail)*)?] $($rest)*);
    };

    (@pool $vis:vis $name:ident [$($off:tt)*]) => {
        $vis const fn $name(i: usize) -> ::concinnity_core::ecs::asset_id::AssetId {
            ::concinnity_core::ecs::asset_id::AssetId(IDS_BASE + ($($off)*) as u32 + i as u32)
        }
    };
}

pub(crate) use hud_ids;

#[cfg(test)]
mod tests {
    use super::*;

    mod sample {
        use super::super::{Family, family_base};

        pub(super) const AREA_SPAN: usize = 4;
        const ROWS: usize = 3;

        hud_ids! {
            base: family_base(Family::Modal);
            sprites: [PANEL_BG, row_bg[ROWS], CLOSE_BG];
            labels: [TITLE, [ROWS] { caption, value }, ..area()];
            code_labels: [line[2]];
            fields: [INPUT = "search", slot[2] = ""];
            blocks: [pub(super) AREA: AREA_SPAN];
        }

        pub(super) fn area() -> Vec<super::AssetId> {
            (0..AREA_SPAN as u32)
                .map(|i| super::AssetId(AREA + i))
                .collect()
        }

        pub(super) fn named() -> [super::AssetId; 9] {
            [
                PANEL_BG,
                row_bg(0),
                CLOSE_BG,
                TITLE,
                caption(2),
                value(0),
                line(1),
                INPUT,
                slot(1),
            ]
        }
    }

    fn base() -> u32 {
        family_base(Family::Modal)
    }

    // Each list keeps its declaration order, and a group's members interleave.
    #[test]
    fn lists_follow_declaration_order() {
        let ids = sample::ids();
        let b = base();
        let at = |n: u32| AssetId(b + n);
        assert_eq!(ids.sprites, [at(0), at(1), at(2), at(3), at(4)]);
        let mut labels = vec![at(5), at(6), at(9), at(7), at(10), at(8), at(11)];
        labels.extend(sample::area());
        assert_eq!(ids.labels, labels);
        assert_eq!(ids.code_labels, [at(12), at(13)]);
        assert_eq!(ids.fields, [(at(14), "search"), (at(15), ""), (at(16), "")]);
    }

    // Names and pool accessors resolve to the ids their lists hold, and a block
    // starts past every id declared before it.
    #[test]
    fn names_and_pools_number_from_the_base() {
        let b = base();
        let n = sample::named();
        let expected = [0, 1, 4, 5, 8, 9, 13, 14, 16].map(|o| AssetId(b + o));
        assert_eq!(n, expected);
        assert_eq!(sample::AREA, b + 17);
    }

    // Every id is listed once, and `all` covers each kind.
    #[test]
    fn all_lists_every_id_once() {
        let all: Vec<AssetId> = sample::ids().all().collect();
        let unique: std::collections::BTreeSet<AssetId> = all.iter().copied().collect();
        assert_eq!(all.len(), unique.len());
        assert_eq!(all.len(), 5 + 7 + sample::AREA_SPAN + 2 + 3);
    }

    #[test]
    fn hide_blanks_and_blurs_every_element() {
        use concinnity_core::components::{Sprite, TextInput, TextLabel};
        let ids = sample::ids();
        let mut world = World::new();
        for &id in &ids.sprites {
            world.push_identified(
                id,
                Sprite {
                    visible: true,
                    ..Default::default()
                },
            );
        }
        for &id in ids.labels.iter().chain(&ids.code_labels) {
            world.push_identified(
                id,
                TextLabel {
                    visible: true,
                    ..Default::default()
                },
            );
        }
        for id in ids.field_ids() {
            world.push_identified(
                id,
                TextInput {
                    visible: true,
                    focused: true,
                    ..Default::default()
                },
            );
        }
        ids.hide(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible && !t.focused));
    }

    // Families never overlap: each starts a full stride past the one before.
    #[test]
    fn families_are_a_stride_apart() {
        assert_eq!(
            family_base(Family::TopBar) + FAMILY_SPAN as u32,
            family_base(Family::Highlight)
        );
        assert_eq!(
            family_base(Family::Loading) + FAMILY_SPAN as u32,
            panel_base(PanelKey::ALL[0])
        );
        assert_eq!(
            panel_base(PanelKey::ALL[0]) + FAMILY_SPAN as u32,
            panel_base(PanelKey::ALL[1])
        );
    }
}

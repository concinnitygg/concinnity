// src/editor/registry.rs
//
// The floating-panel registry. Every editor panel is one `Panel` implementation
// (in `hook/panels.rs`) plus one entry in `PANELS` below; everything that used
// to be hand-wired per panel derives from the registry instead: the reserved-id
// allocation, the View panel's toggle rows, HUD injection (`inject.rs`), the
// focus-stack draw layers, title-bar dragging, close buttons, click / wheel
// routing, and the hidden pass (`hook/routing.rs` / `hook/layout.rs`). Adding a
// panel means a `PanelKey` variant, a `Panel` impl, and its `PANELS` entry --
// none of the shared machinery is touched.

use super::hook::{EditorHook, panels};
use crate::assets::FrameInput;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Base of the editor HUD's reserved asset-id space. Interned world ids are dense
// from 0 and never approach this range, and these ids are never serialized to a
// blob. The top bar (`hud.rs`) owns `ID_BASE + 0x0..0x40`; the panels' families
// are allocated by `base` below.
pub(crate) const ID_BASE: u32 = 0x3000_0000;

// One variant per floating panel, in default back-to-front draw / focus order
// (later = frontmost at launch). The discriminant indexes `PANELS` and the
// hook's per-panel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelKey {
    Assets,
    Edit,
    Preview,
    View,
    Templates,
    Lighting,
    Story,
    Import,
    Health,
    Console,
    Behavior,
    Variables,
    Content,
    // Last (default frontmost), so the detail floats over the Templates list
    // it spawns from before any interaction reorders the focus stack.
    TemplateDetail,
}

pub(crate) const PANEL_COUNT: usize = 14;

impl PanelKey {
    pub(crate) const ALL: [PanelKey; PANEL_COUNT] = [
        PanelKey::Assets,
        PanelKey::Edit,
        PanelKey::Preview,
        PanelKey::View,
        PanelKey::Templates,
        PanelKey::Lighting,
        PanelKey::Story,
        PanelKey::Import,
        PanelKey::Health,
        PanelKey::Console,
        PanelKey::Behavior,
        PanelKey::Variables,
        PanelKey::Content,
        PanelKey::TemplateDetail,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

// Reserved-id base of each panel's element family, allocated here so families
// can never collide (`id_families_are_disjoint` enforces it against every
// panel's actual id list). A new panel takes the next free block; the Assets and
// Edit families predate the 0x100 spacing and span wider.
pub(crate) const fn base(key: PanelKey) -> u32 {
    ID_BASE
        + match key {
            PanelKey::Assets => 0x40,
            PanelKey::Edit => 0x200,
            PanelKey::Preview => 0x400,
            PanelKey::View => 0x500,
            PanelKey::Templates => 0x600,
            PanelKey::TemplateDetail => 0x700,
            PanelKey::Lighting => 0x800,
            PanelKey::Story => 0x900,
            PanelKey::Import => 0xA00,
            PanelKey::Health => 0xB00,
            // 0xC00..0xE00 belong to the highlight, gizmo, and marquee
            // overlays, and 0x1000.. to the billboards (all allocated in
            // their own modules). Above the billboards' open-ended run, which
            // ends well short of 0x2000 today (`id_families_are_disjoint`
            // holds the line).
            PanelKey::Console => 0x2000,
            // The outline, its value column, and the palette give this panel
            // the widest family of any, so it takes a whole block of its own.
            PanelKey::Behavior => 0x3000,
            // Clear of the Behavior panel's whole block, whose chart segments
            // and card pools run well past 0x3300.
            PanelKey::Variables => 0x4000,
            PanelKey::Content => 0x5000,
        }
}

// A floating editor panel. Implementations are stateless units (all panel state
// lives on the hook); the registry consumers drive them, so a panel never wires
// its own dragging, focus, close button, injection, or hidden pass.
pub(crate) trait Panel: Sync {
    // The key this panel is registered under (pinned by `keys_match_registry`).
    fn key(&self) -> PanelKey;
    // Caption of this panel's toggle row in the View panel; `None` keeps it out
    // (panels that open from elsewhere, like the edit form).
    fn view_row(&self) -> Option<&'static str> {
        None
    }
    // Whether the user can resize this panel by dragging its edges / corners.
    // Default `false`: the fixed-size panels (Preview, Health, Console) opt out.
    fn resizable(&self) -> bool {
        false
    }
    // Whether the panel is shown and interactive this frame, compound gates
    // included (e.g. the edit form requires the Assets UI to be on).
    fn is_open(&self, hook: &EditorHook) -> bool;
    // Flip the panel's shown state (its View-panel toggle row). Opening may
    // seed the panel's typed controls from the world, hence the world access.
    fn toggle(&self, _hook: &mut EditorHook, _world: &mut World) {}
    // The title-bar "X".
    fn close(&self, hook: &mut EditorHook, world: &mut World);
    // The panel footprint this frame (it may track dynamic content), for the
    // drag clamp and the shared title-bar geometry. Also the minimum a resizable
    // panel can be dragged to.
    fn size(&self, hook: &EditorHook) -> [f32; 2];
    // The largest a resizable panel may be dragged to, per axis (`f32::INFINITY`
    // for unbounded, capped only by the screen). Default unbounded; a panel whose
    // body is a fixed pool of rows caps its height so it never shows empty space.
    fn max_size(&self, _hook: &EditorHook) -> [f32; 2] {
        [f32::INFINITY, f32::INFINITY]
    }
    // Where the panel sits until the user drags it.
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2];
    // The injected element ids: sprites, labels, and typed fields with their
    // placeholder text. The source for HUD injection and the draw-layer map.
    fn sprite_ids(&self) -> Vec<AssetId>;
    fn label_ids(&self) -> Vec<AssetId>;
    fn field_ids(&self) -> Vec<(AssetId, &'static str)> {
        Vec::new()
    }
    // The elements of a floating overlay the panel currently has open (a
    // palette, a dropdown): they draw above the rest of the panel, so an opaque
    // backing occludes what it covers instead of the covered text showing
    // through it. Empty while no overlay is open.
    fn overlay_ids(&self, _hook: &EditorHook) -> Vec<AssetId> {
        Vec::new()
    }
    // Resolve + apply a body press at `(mx, my)` for the panel at origin `o`;
    // the title bar and close button never reach this. `false` lets the press
    // fall through to the panel behind.
    fn press(
        &self,
        hook: &mut EditorHook,
        world: &mut World,
        mx: f32,
        my: f32,
        o: [f32; 2],
    ) -> bool;
    // Whether a wheel at `(mx, my)` lands in this panel's scrollable region.
    fn wheel_over(
        &self,
        _hook: &EditorHook,
        _world: &World,
        _mx: f32,
        _my: f32,
        _o: [f32; 2],
    ) -> bool {
        false
    }
    // Move the panel's scroll region one step in the wheel direction.
    fn scroll(&self, _hook: &mut EditorHook, _world: &mut World, _delta: f32) {}
    // Per-frame editing keys (`FrameInput.captured_key`), delivered to the
    // frontmost open panel only, so panels never fight over the keyboard.
    fn frame_keys(&self, _hook: &mut EditorHook, _world: &mut World, _input: &FrameInput) {}
    // Per-frame layout while shown.
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]);
    // Blank every element (toggled off, or the F1-hidden pass).
    fn hide(&self, world: &mut World);
}

// The registered panels, indexed by `PanelKey`.
static PANELS: [&dyn Panel; PANEL_COUNT] = [
    &panels::AssetsPanel,
    &panels::EditPanel,
    &panels::PreviewPanel,
    &panels::ViewPanel,
    &panels::TemplatesPanel,
    &panels::LightingPanel,
    &panels::StoryPanel,
    &panels::ImportPanel,
    &panels::HealthPanel,
    &panels::ConsolePanel,
    &panels::BehaviorPanel,
    &panels::VariablesPanel,
    &panels::ContentPanel,
    &panels::TemplateDetailPanel,
];

pub(crate) fn panel(key: PanelKey) -> &'static dyn Panel {
    PANELS[key.index()]
}

// Every registered panel, in registry (back-to-front) order.
pub(crate) fn all() -> impl Iterator<Item = &'static dyn Panel> {
    PANELS.into_iter()
}

// The panels listed in the View panel, in registry order (row `i` of the View
// panel toggles the `i`-th entry here).
pub(crate) fn view_toggles() -> impl Iterator<Item = &'static dyn Panel> {
    PANELS.into_iter().filter(|p| p.view_row().is_some())
}

pub(crate) fn view_toggle_count() -> usize {
    view_toggles().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every registry slot holds the panel it is indexed as, so `panel(key)`
    // can never route one panel's input to another.
    #[test]
    fn keys_match_registry() {
        for key in PanelKey::ALL {
            assert_eq!(panel(key).key(), key);
        }
    }

    // Every reserved id across every panel (plus the top bar) is unique: the
    // point of allocating the family bases in one place.
    #[test]
    fn id_families_are_disjoint() {
        let mut seen: std::collections::HashMap<AssetId, String> = std::collections::HashMap::new();
        let mut claim = |id: AssetId, who: String| {
            if let Some(prev) = seen.insert(id, who.clone()) {
                panic!("{who} and {prev} both claim {id:?}");
            }
        };
        for id in super::super::hud::all_ids() {
            claim(id, "top bar".to_string());
        }
        for id in super::super::highlight::all_sprite_ids() {
            claim(id, "selection highlight".to_string());
        }
        claim(super::super::marquee::RECT, "marquee rect".to_string());
        for id in super::super::gizmo::all_sprite_ids() {
            claim(id, "gizmo".to_string());
        }
        for id in super::super::billboards::all_sprite_ids() {
            claim(id, "billboards".to_string());
        }
        for id in super::super::billboards::all_label_ids() {
            claim(id, "billboard glyphs".to_string());
        }
        claim(
            super::super::gizmo::MODE_LABEL,
            "gizmo mode label".to_string(),
        );
        claim(super::super::cursor::CURSOR, "editor cursor".to_string());
        for id in super::super::create_menu::all_sprite_ids() {
            claim(id, "create menu sprites".to_string());
        }
        for id in super::super::create_menu::all_label_ids() {
            claim(id, "create menu labels".to_string());
        }
        for id in super::super::view_menu::all_sprite_ids() {
            claim(id, "display menu sprites".to_string());
        }
        for id in super::super::view_menu::all_label_ids() {
            claim(id, "display menu labels".to_string());
        }
        for key in PanelKey::ALL {
            let p = panel(key);
            for id in p.sprite_ids() {
                claim(id, format!("{key:?} sprites"));
            }
            for id in p.label_ids() {
                claim(id, format!("{key:?} labels"));
            }
            for (id, _) in p.field_ids() {
                claim(id, format!("{key:?} fields"));
            }
        }
    }

    // The View panel's toggle rows come from the registry in registry order.
    #[test]
    fn view_toggles_lists_the_toggleable_panels() {
        let rows: Vec<&'static str> = view_toggles().map(|p| p.view_row().unwrap()).collect();
        assert_eq!(
            rows,
            [
                "Assets",
                "Preview",
                "Templates",
                "Lighting",
                "Story",
                "Import",
                "Health",
                "Console",
                "Behavior",
                "Variables",
                "Content"
            ]
        );
    }
}

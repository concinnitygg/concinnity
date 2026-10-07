//! The floating-panel registry. Every editor panel is one `Panel` implementation
//! (in `hook/panels.rs`) plus one entry in the `panels!` list below; everything
//! that used to be hand-wired per panel derives from the registry instead: the
//! reserved-id family (`hud_ids.rs`), the View panel's toggle rows, HUD
//! injection (`inject.rs`), the focus-stack draw layers, title-bar dragging,
//! close buttons, click / wheel routing, open state, and the hidden pass
//! (`hook/routing.rs` / `hook/layout.rs`). Adding a panel means a `Panel` impl
//! and its entry in the list -- none of the shared machinery is touched.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use crate::editor::hook::{EditorHook, panels};
use crate::editor::hud_ids::HudIds;

// Declares `PanelKey` (one variant per entry, its discriminant indexing the
// panel table and the hook's per-panel state), `PanelKey::ALL`, `PANEL_COUNT`,
// the table itself, and `VIEW_ROWS`, all from the one list below. An entry with
// a caption gets a toggle row in the View panel.
macro_rules! panels {
    ($($key:ident $(($caption:literal))? => $panel:expr,)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum PanelKey {
            $($key,)+
        }

        pub(crate) const PANEL_COUNT: usize = [$(PanelKey::$key),+].len();

        impl PanelKey {
            pub(crate) const ALL: [PanelKey; PANEL_COUNT] = [$(PanelKey::$key),+];
        }

        static PANELS: [&dyn Panel; PANEL_COUNT] = [$(&$panel),+];

        // The View panel's toggle rows, in registry order (row `i` toggles the
        // `i`-th entry), with their captions.
        pub(crate) const VIEW_ROWS: &[(PanelKey, &str)] = &[$($((PanelKey::$key, $caption),)?)+];
    };
}

// Every floating panel, in default back-to-front draw / focus order (later =
// frontmost at launch).
panels! {
    Assets("Assets") => panels::AssetsPanel,
    Edit => panels::EditPanel,
    Preview("Preview") => panels::PreviewPanel,
    View => panels::ViewPanel,
    Templates("Templates") => panels::TemplatesPanel,
    Lighting("Lighting") => panels::LightingPanel,
    Story("Story") => panels::StoryPanel,
    Shaders("Shaders") => panels::ShadersPanel,
    Import("Import") => panels::ImportPanel,
    Health("Health") => panels::HealthPanel,
    Console("Console") => panels::ConsolePanel,
    Behavior("Behavior") => panels::BehaviorPanel,
    Map("Map") => panels::MapPanel,
    Variables("Variables") => panels::VariablesPanel,
    Content("Content") => panels::ContentPanel,
    CharacterShape("Character Shape") => panels::CharacterShapePanel,
    // Late (default frontmost), so the detail floats over the Templates list
    // it spawns from before any interaction reorders the focus stack.
    TemplateDetail => panels::TemplateDetailPanel,
    // Likewise over the Shaders list it opens from.
    ShaderSource => panels::ShaderSourcePanel,
    // The palette is a transient launcher overlay, so it starts above
    // everything it can open.
    Palette => panels::PalettePanel,
    // Last of all: a session started with no world named on the command line
    // opens on this panel, so it has to be the frontmost thing on screen.
    Worlds("Worlds") => panels::WorldsPanel,
}

impl PanelKey {
    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

// One value per panel, indexed by `PanelKey`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PerPanel<T>([T; PANEL_COUNT]);

impl<T: Copy> PerPanel<T> {
    pub(crate) const fn splat(value: T) -> Self {
        Self([value; PANEL_COUNT])
    }
}

impl<T> std::ops::Index<PanelKey> for PerPanel<T> {
    type Output = T;
    fn index(&self, key: PanelKey) -> &T {
        &self.0[key.index()]
    }
}

impl<T> std::ops::IndexMut<PanelKey> for PerPanel<T> {
    fn index_mut(&mut self, key: PanelKey) -> &mut T {
        &mut self.0[key.index()]
    }
}

// A floating editor panel. Implementations are stateless units (all panel state
// lives on the hook); the registry consumers drive them, so a panel never wires
// its own dragging, focus, close button, injection, or hidden pass.
pub(crate) trait Panel: Sync {
    // The key this panel is registered under (pinned by `keys_match_registry`).
    fn key(&self) -> PanelKey;
    // The injected element ids, for HUD injection, the draw-layer map and the
    // hidden pass.
    fn ids(&self) -> &'static HudIds;
    // Whether the user can resize this panel by dragging its edges / corners.
    // Default `false`: the fixed-size panels (Preview, Health, Console) opt out.
    fn resizable(&self) -> bool {
        false
    }
    // Whether the panel is shown and interactive this frame. A panel whose
    // showing follows from other state (e.g. the edit form requires the Assets
    // UI to be on) overrides this.
    fn is_open(&self, hook: &EditorHook) -> bool {
        hook.open_flag(self.key())
    }
    // Flip the panel's shown state (its View-panel toggle row).
    fn toggle(&self, hook: &mut EditorHook, world: &mut World) {
        let open = !hook.open_flag(self.key());
        hook.set_open_flag(self.key(), open);
        if open {
            self.on_open(hook, world);
        }
    }
    // Runs when the toggle opens the panel. Opening may seed the panel's typed
    // controls from the world, hence the world access.
    fn on_open(&self, _hook: &mut EditorHook, _world: &mut World) {}
    // The title-bar "X".
    fn close(&self, hook: &mut EditorHook, world: &mut World) {
        hook.set_open_flag(self.key(), false);
        self.on_close(hook, world);
    }
    // Runs when the title-bar "X" closes the panel.
    fn on_close(&self, _hook: &mut EditorHook, _world: &mut World) {}
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
    // The wheel at `(mx, my)`, for a panel with more than one region to
    // scroll; the rest scroll their one region.
    fn scroll_at(&self, hook: &mut EditorHook, world: &mut World, delta: f32, _mx: f32, _my: f32) {
        self.scroll(hook, world, delta);
    }
    // Per-frame editing keys (`FrameInput.key_events`), delivered to the
    // frontmost open panel only, so panels never fight over the keyboard.
    fn frame_keys(&self, _hook: &mut EditorHook, _world: &mut World, _input: &FrameInput) {}
    // Per-frame layout while shown.
    fn draw(&self, hook: &EditorHook, world: &mut World, o: [f32; 2], mouse: [f32; 2]);
    // Blank every element (toggled off, or the F1-hidden pass).
    fn hide(&self, world: &mut World) {
        self.ids().hide(world);
    }
}

pub(crate) fn panel(key: PanelKey) -> &'static dyn Panel {
    PANELS[key.index()]
}

// Every registered panel, in registry (back-to-front) order.
pub(crate) fn all() -> impl Iterator<Item = &'static dyn Panel> {
    PANELS.into_iter()
}

// The panels listed in the View panel, in `VIEW_ROWS` order.
pub(crate) fn view_toggles() -> impl Iterator<Item = &'static dyn Panel> {
    VIEW_ROWS.iter().map(|&(key, _)| panel(key))
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

    // Every reserved id across every panel and every other HUD module is
    // unique, so no two elements ever fight over one.
    #[test]
    fn id_families_are_disjoint() {
        use crate::editor::{create_menu, hud, modal, toast_overlay, view_menu, viewport, worlds};
        let mut seen: std::collections::HashMap<AssetId, String> = std::collections::HashMap::new();
        let mut claim = |who: String, ids: &mut dyn Iterator<Item = AssetId>| {
            for id in ids {
                if let Some(prev) = seen.insert(id, who.clone()) {
                    panic!("{who} and {prev} both claim {id:?}");
                }
            }
        };
        let singles = [
            ("marquee", viewport::marquee::RECT),
            ("editor cursor", viewport::cursor::CURSOR),
            ("start screen shot fade", worlds::cinematic::FADE),
        ];
        for (who, id) in singles {
            claim(who.to_string(), &mut std::iter::once(id));
        }
        let others = [
            ("top bar", hud::ids()),
            ("selection highlight", viewport::highlight::ids()),
            ("gizmo", viewport::gizmo::ids()),
            ("billboards", viewport::billboards::ids()),
            ("create menu", create_menu::ids()),
            ("display menu", view_menu::ids()),
            ("toasts", toast_overlay::ids()),
            ("confirm dialog", modal::ids()),
            ("loading cover", worlds::loading::ids()),
        ];
        for (who, ids) in others {
            claim(who.to_string(), &mut ids.all());
        }
        for key in PanelKey::ALL {
            claim(format!("{key:?}"), &mut panel(key).ids().all());
        }
    }

    // Each key reads and writes its own slot.
    #[test]
    fn per_panel_indexes_by_key() {
        let mut open = PerPanel::splat(false);
        open[PanelKey::Map] = true;
        for key in PanelKey::ALL {
            assert_eq!(open[key], key == PanelKey::Map, "{key:?}");
        }
    }

    // The View panel's toggle rows come from the registry in registry order.
    #[test]
    fn view_toggles_lists_the_toggleable_panels() {
        let rows: Vec<&str> = VIEW_ROWS.iter().map(|&(_, caption)| caption).collect();
        assert_eq!(
            rows,
            [
                "Assets",
                "Preview",
                "Templates",
                "Lighting",
                "Story",
                "Shaders",
                "Import",
                "Health",
                "Console",
                "Behavior",
                "Map",
                "Variables",
                "Content",
                "Character Shape",
                "Worlds"
            ]
        );
    }
}

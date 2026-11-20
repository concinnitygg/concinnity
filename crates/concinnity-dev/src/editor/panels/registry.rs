//! The floating-panel registry. Every editor panel is one `Panel` implementation
//! (in `hook/panels.rs`) plus one entry in the `panels!` list below; what is
//! shared derives from the registry: the View panel's toggle rows, building the
//! panels into a world, the focus-stack draw layers, close buttons, and open
//! state. Each panel is built from engine widgets, which move, size, and drive
//! it. Adding a panel means a `Panel` impl and its entry in the list -- none of
//! the shared machinery is touched.

use concinnity_engine::components::{FrameInput, UiEvent};
use concinnity_engine::ecs::World;

use crate::editor::hook::{EditorHook, panels};
use crate::editor::ui::{BuiltPanel, UiBuilder};

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
    Planet("Planet") => panels::PlanetPanel,
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

impl<T: Default> Default for PerPanel<T> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| T::default()))
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
// its own moving, sizing, focus, close button, or building.
pub(crate) trait Panel: Sync {
    // The key this panel is registered under (pinned by `keys_match_registry`).
    fn key(&self) -> PanelKey;
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
    // The panel's footprint this frame (it may track dynamic content), which
    // is also the least a resizable panel can be sized to.
    fn size(&self, hook: &EditorHook) -> [f32; 2];
    // Where the panel sits until the user drags it.
    fn default_origin(&self, vp: [f32; 2]) -> [f32; 2];
    // Per-frame editing keys (`FrameInput.key_events`), delivered to the
    // frontmost open panel only, so panels never fight over the keyboard.
    fn frame_keys(&self, _hook: &mut EditorHook, _world: &mut World, _input: &FrameInput) {}
    // Build the panel into a world the editor is setting up.
    fn build(&self, ui: &mut UiBuilder<'_>) -> Option<BuiltPanel>;
    // Take in what the user did to the panel's widgets since last frame that
    // no event reports, such as a field the press focused.
    fn observe(&self, _hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {}
    // Give a world the panel was just built into what the user typed into
    // its fields in the world before.
    fn restore(&self, _hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {}
    // Bring what the hook derives for the panel up to date and write it into
    // the world's UI state and the elements of `built` it changes, each frame
    // it shows.
    fn sync(&self, _hook: &mut EditorHook, _world: &mut World, _built: &mut BuiltPanel) {}
    // Lend the world's widgets what the hook keeps across world rebuilds, such
    // as a text area's editor, after `sync`; `true` when something was lent,
    // so `reclaim` takes it back before the hook works with it again.
    fn lend(&self, _hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) -> bool {
        false
    }
    // Take back what `lend` gave the world's widgets.
    fn reclaim(&self, _hook: &mut EditorHook, _world: &mut World, _built: &BuiltPanel) {}
    // Handle an event the panel's widgets sent; `false` when it is not one of
    // the panel's.
    fn on_event(&self, _hook: &mut EditorHook, _world: &mut World, _event: &UiEvent) -> bool {
        false
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
                "Planet",
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

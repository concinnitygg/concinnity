//! Setting the editor up in an already-compiled world: its window pinned to a
//! usable shape, the resources its tools read (viewport picking, thumbnails,
//! sandboxed saves), the engine's debug HUD handed over to it, and its whole
//! UI built from engine widgets on one persistent window-space screen. Runs
//! between the world's in-memory compile and `Runtime::start`, and again on
//! every live-preview rebuild; none of it is written back to the user's
//! world.jsonl (SAVE serializes the authored entry list, not the live world).

use concinnity_engine::components::{
    DebugHud, EngineDefaults, Screen, ScreenInput, ScreenSpace, Window, WindowMode,
};
use concinnity_engine::ecs::{FontHandle, World};
use concinnity_engine::protocol::{PickIndex, TransientSaves};

use super::panels::registry::{self, PanelKey};
use super::ui::{BuiltUi, UiBuilder};

// The editor's screen draws over, and takes input over, any screen the world
// opens.
const EDITOR_LAYER: i32 = 1000;

// Reuse the engine HUD font for the editor's text: baked here rather than at
// world start so the panels have a handle before the completion pass runs.
// Sharing it keeps a user world's display font from restyling the editor, and
// costs one atlas rather than two. A bake failure leaves the panels on the
// built-in face: a mis-sized HUD still beats an invisible one.
fn hud_font(world: &mut World) -> Option<FontHandle> {
    concinnity_engine::defaults::hud_font(&mut world.context()).ok()
}

// Set the editor up in `world`.
pub(crate) fn editor_ui(world: &mut World) {
    let font = hud_font(world);
    let code_font = super::code_font::inject(world);
    pin_editor_window(world);
    // Opt in to viewport picking: GraphicsSystem captures pick candidates at
    // start only when this resource is already present, then refreshes it each
    // frame with world-space AABBs.
    world.insert_resource(PickIndex::default());
    // Baked asset thumbnails join the sprite atlas pool at graphics init; the
    // Content panel's cells sample them by reserved handle. Captured on every
    // setup so a rebuild picks up freshly baked images.
    world.insert_resource(super::thumbs::capture_for_setup());
    // An editor session never touches the user's real save files: the systems
    // that persist play state sample this at init and sandbox their saves, so
    // every preview run starts fresh (see the protocol type).
    world.insert_resource(TransientSaves(true));
    // The editor replaces the engine's debug HUD (both would answer F1): drop
    // one the world declares and turn the default one off, so `World::start`
    // neither completes the world with one nor constructs its system.
    world.remove_all::<DebugHud>();
    match world.query_mut::<EngineDefaults>().next() {
        Some(declared) => declared.debug_hud = false,
        None => world.add_component(EngineDefaults {
            debug_hud: false,
            ..Default::default()
        }),
    }
    build(world, font, code_font);
}

// Build everything the editor draws onto its own screen, in draw order: the
// viewport's marks under every panel, the panels, then the overlays above them.
fn build(world: &mut World, font: Option<FontHandle>, code_font: Option<FontHandle>) {
    let Ok((screen, _)) = world.context().push_minted(Screen {
        persistent: true,
        layer: EDITOR_LAYER,
        space: ScreenSpace::Window,
        input: ScreenInput::Passthrough,
        pauses_world: false,
        ..Default::default()
    }) else {
        return;
    };
    let mut built = BuiltUi {
        fresh: true,
        ..Default::default()
    };
    let mut ui = UiBuilder::new(world, screen, font).with_code_font(code_font);
    built.marks = super::viewport::marks::build(&mut ui);
    for key in PanelKey::ALL {
        built.panels[key] = registry::panel(key).build(&mut ui);
    }
    built.top_bar = super::hud::build(&mut ui);
    built.dialog = super::modal::build(&mut ui);
    built.toasts = super::toasts::build(&mut ui);
    built.display_menu = super::view_menu::build(&mut ui);
    built.create_menu = super::create_menu::build(&mut ui);
    built.cursor = super::viewport::cursor::build(&mut ui);
    world.insert_resource(built);
}

// Whether the editor drops its window's title bar. macOS keeps the close /
// minimize / zoom buttons floating over the content without one, so the window
// stays movable and closable. Windows and Linux draw those controls inside the
// title bar and leave no drag handle behind it, and the editor's own top bar is
// not a substitute (no close button, and no OS-level caption hit-testing), so
// the editor keeps its chrome there rather than strand a window the user cannot
// move or close.
const HIDES_TITLE_BAR: bool = cfg!(target_os = "macos");

// Pin the editor's own window to a usable shape: always windowed, and (where the
// chrome survives, see `HIDES_TITLE_BAR`) title-bar-less. A `borderless` or
// `fullscreen` world would otherwise size the editor window to the whole display,
// so the in-engine cursor's inside-the-content test is true everywhere and the OS
// cursor stays hidden past the panels; forcing windowed keeps the OS pointer for
// the desktop while the in-engine arrow (and its resize shape) owns the viewport.
// Overriding the live component rather than the authored entry keeps it out of
// the user's world.jsonl on SAVE, so the authored `mode` still ships to `cn run`
// and `cn export`; being part of this one setup seam means every live-preview
// rebuild re-applies it, so a rebuilt world cannot restore the authored mode or
// title bar on the next window restyle.
fn pin_editor_window(world: &mut World) {
    let mut authored = false;
    for w in world.query_mut::<Window>() {
        w.mode = WindowMode::Windowed;
        if HIDES_TITLE_BAR {
            w.title_bar = false;
        }
        authored = true;
    }
    // A world with no Window of its own renders through `Window::default()`;
    // materialize that default so the overrides have something to sit on. The
    // default is already windowed, so only the title bar needs pinning here.
    if !authored && HIDES_TITLE_BAR {
        world.add_component(Window {
            title_bar: false,
            ..Default::default()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{Button, Identity, Sprite, TextField, TextLabel};
    use concinnity_engine::ecs::Ref;
    use concinnity_engine::ecs::asset_id::AssetId;
    use concinnity_engine::resource::FontTable;

    // The editor draws its own chrome, so setup turns the world's title bar
    // off in place -- keeping the rest of the authored window untouched. Where
    // the title bar is the window's only drag handle and close button, the
    // authored value stands instead.
    #[test]
    fn setup_overrides_an_authored_title_bar_only_where_the_chrome_survives() {
        let mut world = World::new();
        world.add_component(Window {
            title: "My Game".into(),
            width: 1600,
            title_bar: true,
            ..Default::default()
        });
        editor_ui(&mut world);

        let w = world.query::<Window>().next().unwrap();
        assert_eq!(w.title_bar, !HIDES_TITLE_BAR);
        // Only the chrome is the editor's business; size and title stay authored.
        assert_eq!(w.title, "My Game");
        assert_eq!(w.width, 1600);
        assert_eq!(world.query::<Window>().count(), 1);
    }

    #[test]
    fn setup_materializes_a_window_to_override_only_where_the_chrome_survives() {
        let mut world = World::new();
        editor_ui(&mut world);

        let windows: Vec<&Window> = world.query::<Window>().collect();
        if HIDES_TITLE_BAR {
            assert_eq!(windows.len(), 1);
            assert!(!windows[0].title_bar);
        } else {
            assert!(windows.is_empty());
        }
    }

    // A `borderless` / `fullscreen` world would size the editor window to the
    // whole display and swallow the OS cursor past the panels, so setup pins
    // the editor session to windowed on every platform -- while leaving the rest
    // of the authored window (size, title) for `cn run` / `cn export` to honor.
    #[test]
    fn setup_pins_a_non_windowed_mode_to_windowed() {
        for mode in [WindowMode::Borderless, WindowMode::Fullscreen] {
            let mut world = World::new();
            world.add_component(Window {
                title: "My Game".into(),
                width: 1600,
                mode,
                ..Default::default()
            });
            editor_ui(&mut world);

            let w = world.query::<Window>().next().unwrap();
            assert_eq!(
                w.mode,
                WindowMode::Windowed,
                "editor pins {mode:?} to windowed"
            );
            // Only the mode (and chrome) is the editor's business; the rest stands.
            assert_eq!(w.title, "My Game");
            assert_eq!(w.width, 1600);
            assert_eq!(world.query::<Window>().count(), 1);
        }
    }

    // The button + field text draws with the engine HUD face, baked into the
    // world's font table here rather than at world start -- NOT the first
    // label in the world, whose font may be a user world's oversized display
    // face. Code text draws with the monospace face appended after it. The
    // DebugHud is dropped and its default turned off, since the editor takes
    // over F1.
    #[test]
    fn bakes_the_hud_and_code_faces_and_takes_over_the_debug_hud() {
        let mut world = World::new();
        world.push_identified(
            AssetId(0),
            TextLabel {
                font: Some(FontHandle::new(99)),
                ..Default::default()
            },
        );
        world.add_component(DebugHud::default());
        editor_ui(&mut world);

        // Two faces appended: the HUD face the panels use, then the code face.
        let fonts = world.resource::<FontTable>().expect("the faces were baked");
        assert_eq!(fonts.len(), 2);
        let baked = FontHandle::new(0);
        let code = world
            .query::<concinnity_engine::components::TextArea>()
            .next()
            .unwrap();
        assert_eq!(code.style.font, Some(FontHandle::new(1)));
        let save = world.query::<Button>().find(|b| b.label == "Save").unwrap();
        assert_eq!(save.style.font, Some(baked));
        let field = world.query::<TextField>().next().unwrap();
        assert_eq!(field.font, Some(baked));

        assert_eq!(world.query::<DebugHud>().count(), 0, "DebugHud dropped");
        let defaults = world
            .query::<EngineDefaults>()
            .next()
            .expect("the debug HUD default is turned off");
        assert!(!defaults.debug_hud);
        assert!(defaults.hud, "every other default is left alone");
    }

    // A world that already declares the directive keeps its own flags; only the
    // debug HUD is forced off, and no second directive appears (two would fail
    // the world start).
    #[test]
    fn a_declared_directive_is_amended_rather_than_duplicated() {
        let mut world = World::new();
        world.add_component(EngineDefaults {
            loading_overlay: false,
            ..Default::default()
        });
        editor_ui(&mut world);

        assert_eq!(world.query::<EngineDefaults>().count(), 1);
        let defaults = world.query::<EngineDefaults>().next().unwrap();
        assert!(!defaults.debug_hud);
        assert!(!defaults.loading_overlay, "the world's own opt-out stands");
    }

    // Everything the editor builds belongs to its one screen, and the world's
    // own elements are left for the world's HUD.
    #[test]
    fn every_editor_element_belongs_to_the_editor_screen() {
        let mut world = World::new();
        world.add_component(TextLabel::default());
        editor_ui(&mut world);

        let screens: Vec<(AssetId, &Screen)> = world
            .join2::<Screen, Identity>()
            .map(|(_, s, id)| (id.id(), s))
            .collect();
        assert_eq!(screens.len(), 1);
        let (id, screen) = screens[0];
        assert!(screen.persistent && screen.space == ScreenSpace::Window);
        let editor = Some(Ref::new(id));
        assert!(world.query::<Sprite>().all(|s| s.screen == editor));
        assert!(world.query::<TextField>().all(|f| f.screen == editor));
        let labels: Vec<&TextLabel> = world.query::<TextLabel>().collect();
        assert_eq!(
            labels.iter().filter(|l| l.screen == editor).count(),
            labels.len() - 1,
            "the world's own label is left for the world's HUD"
        );
    }

    // The top bar comes up with its chips, and every element is identified, so
    // every lookup the tick makes by id resolves.
    #[test]
    fn builds_the_top_bar_and_identifies_every_element() {
        let mut world = World::new();
        editor_ui(&mut world);
        assert!(world.resource::<BuiltUi>().unwrap().top_bar.is_some());
        let labels: Vec<String> = world.query::<Button>().map(|b| b.label.clone()).collect();
        for chip in ["Save", "View", "Display", "Undo", "Redo", "Step", "Stop"] {
            assert!(labels.iter().any(|l| l == chip), "{chip} missing");
        }
        assert_eq!(
            world.join2::<Sprite, Identity>().count(),
            world.query::<Sprite>().count()
        );
        assert_eq!(
            world.join2::<TextField, Identity>().count(),
            world.query::<TextField>().count()
        );
    }
}

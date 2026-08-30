// src/editor/inject.rs
//
// Runtime injection of the editor HUD elements into an already-compiled world.
// Runs between `App::load_blob` and `App::start`, so the injected components are
// indistinguishable from cooked ones -- and none of it is ever written back to
// the user's world.jsonl or blobs (the SAVE path serializes the authored entry
// list, not the live world). The elements are plain `Sprite` / `TextLabel` /
// `TextInput` components at reserved ids; the editor's `DebugHook` tick drives
// them each frame (see `hud.rs` / `panel.rs`). No editor-specific component or
// system is involved, so nothing here reaches the shipped runtime. (The two
// `TextInput` fields do bring in the engine's general text-input system, which
// is real runtime code, not editor-only.)

use super::hud;
use super::registry::{self, PanelKey};
use super::theme;
use crate::components::{
    DebugHud, EngineDefaults, Sprite, TextInput, TextLabel, Window, WindowMode,
};
use crate::ecs::FontHandle;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_cook::authoring::spec::{AssetSpec, asset};

// Placeholder layout used only for frame 0; the tick re-anchors everything to
// the true window corner from the first frame's viewport (`hud::layout`).
const REF_W: f32 = 1280.0;

// Reuse the engine HUD font for the button + field text: the same face the
// injected HUD chips draw with, baked here rather than at world start so the
// panels have a handle before the completion pass runs. Sharing it keeps a
// user world's display font (e.g. the oversized default-font fallback on a
// font-less label) from restyling the editor, and costs one atlas rather than
// two. A bake failure leaves the panels on the built-in face: a mis-sized HUD
// still beats an invisible one.
fn hud_font(world: &mut World) -> Option<FontHandle> {
    concinnity_core::defaults::hud_font(&mut world.context()).ok()
}

// Inject the editor HUD: every registered floating panel and the top bar
// (SAVE / View), all view-less window-space overlays. Injected once by the
// caller, before start. Each panel's elements come from its registry id lists
// and start hidden with placeholder geometry; the hook's tick positions and
// shows only what each frame needs. The overlay draws components in insertion
// order, so the panels go in FIRST (registry order, back-to-front) and the top
// bar LAST: a panel dragged under the bar slides behind it, matching the hook's
// hit-test priority (top bar first). (The hook also publishes a per-frame
// focus-stack draw layer that reorders overlapping panels; the injection order
// is the fallback when the layer map is empty.)
pub(crate) fn editor_hud(world: &mut World) {
    let font = hud_font(world);
    pin_editor_window(world);
    // Opt in to viewport picking: GraphicsSystem captures pick candidates at
    // start only when this resource is already present, then refreshes it each
    // frame with world-space AABBs. Runs on every injection, so a live-preview
    // rebuild keeps the index alive.
    world.insert_resource(crate::ecs::PickIndex::default());
    // Baked asset thumbnails join the sprite atlas pool at graphics init; the
    // Content panel's cell sprites sample them by reserved handle. Captured on
    // every injection so a rebuild picks up freshly baked images.
    world.insert_resource(super::thumbs::capture_for_injection());
    // An editor session never touches the user's real save files: the systems
    // that persist play state sample this at init and sandbox their saves, so
    // every preview run starts fresh (see the protocol type).
    world.insert_resource(crate::ecs::TransientSaves(true));
    // The editor HUD replaces the engine's debug HUD (both would answer F1):
    // drop one the world declares and turn the injected one off, so
    // `World::start` neither completes the world with one nor constructs its
    // system.
    world.remove_all::<DebugHud>();
    match world.query_mut::<EngineDefaults>().next() {
        Some(declared) => declared.debug_hud = false,
        None => world.add_component(EngineDefaults {
            debug_hud: false,
            ..Default::default()
        }),
    }
    // The billboards, selection rings, marquee rect, and gizmo go in first:
    // overlay fallback draw order is insertion order, so they stay under
    // every panel and the top bar even before the per-frame HudLayers publish
    // (the rings draw over the billboards, the gizmo over the rings).
    for s in super::billboards::sprites() {
        world.add_component(s);
    }
    for id in super::billboards::all_label_ids() {
        let mut l = centered_label(id, "", [0.0; 4], font);
        l.visible = false;
        world.add_component(l);
    }
    for s in super::highlight::outline_sprites() {
        world.add_component(s);
    }
    world.add_component(super::marquee::rect_sprite());
    for s in super::gizmo::sprites() {
        world.add_component(s);
    }
    // The editor's in-engine mouse cursor (a follow_cursor sprite): the tick
    // shows it while the editor owns the pointer and drives its resize shape.
    world.add_component(super::cursor::sprite());
    world.add_component(row_label(
        super::gizmo::MODE_LABEL,
        "",
        [0.0, 0.0, 0.0, 0.0],
        font,
        false,
    ));
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        for id in p.sprite_ids() {
            world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
        }
        for id in p.label_ids() {
            world.add_component(row_label(id, "", hidden, font, false));
        }
        for (id, placeholder) in p.field_ids() {
            world.add_component(text_field(id, placeholder, font));
        }
    }
    // The right-click create menu floats over the panels (its per-frame layer
    // also pins it above them while open).
    for id in super::create_menu::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in super::create_menu::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
    // The Display menu floats over the panels the same way.
    for id in super::view_menu::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in super::view_menu::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
    inject_top_bar(world, font);
    // The toast stack goes in after the top bar: it draws over everything (its
    // per-frame layer also pins it there while live).
    for id in super::toast_overlay::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in super::toast_overlay::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
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
// and `cn export`; being part of this one injection seam means every live-preview
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

fn inject_top_bar(world: &mut World, font: Option<FontHandle>) {
    let bar = hud::layout(REF_W);

    world.add_component(button_sprite(
        hud::BAR_BG,
        [0.0, 0.0, REF_W, hud::BAR_H],
        theme::CHROME_TINT,
        true,
    ));
    for (id, rect) in [
        (hud::SAVE_BUTTON, bar.save),
        (hud::VIEW_BUTTON, bar.view),
        (hud::DISPLAY_BUTTON, bar.display),
        (hud::UNDO_BUTTON, bar.undo),
        (hud::REDO_BUTTON, bar.redo),
        (hud::PLAY_BUTTON, bar.play),
        (hud::STEP_BUTTON, bar.step),
        (hud::STOP_BUTTON, bar.stop),
    ] {
        world.add_component(button_sprite(id, rect, theme::BUTTON_TINT, true));
    }
    world.add_component(centered_label(hud::SAVE_LABEL, "Save", bar.save, font));
    world.add_component(centered_label(hud::VIEW_LABEL, "View", bar.view, font));
    world.add_component(centered_label(
        hud::DISPLAY_LABEL,
        "Display",
        bar.display,
        font,
    ));
    world.add_component(centered_label(hud::UNDO_LABEL, "Undo", bar.undo, font));
    world.add_component(centered_label(hud::REDO_LABEL, "Redo", bar.redo, font));
    world.add_component(centered_label(hud::PLAY_LABEL, "Play", bar.play, font));
    world.add_component(centered_label(hud::STEP_LABEL, "Step", bar.step, font));
    world.add_component(centered_label(hud::STOP_LABEL, "Stop", bar.stop, font));
}

// Materialize a templates asset spec into a live component through the engine's
// own accept path (serde over the spec's args) -- the same conversion the cook
// pipeline runs on a world line. The reserved `asset_id` and the reused font are
// set by the caller afterward (neither is part of the spec's args).
fn materialize<T: serde::de::DeserializeOwned>(spec: AssetSpec) -> T {
    serde_json::from_value(crate::spec_args(&spec))
        .expect("editor HUD spec deserializes into its component")
}

fn button_sprite(id: AssetId, rect: [f32; 4], tint: [f32; 4], visible: bool) -> Sprite {
    let mut s: Sprite = materialize(asset::sprite("", rect, tint).set("visible", visible));
    s.asset_id = id;
    s
}

fn centered_label(
    id: AssetId,
    content: &str,
    rect: [f32; 4],
    font: Option<FontHandle>,
) -> TextLabel {
    let pos = [rect[0] + rect[2] * 0.5, rect[1] + hud::LABEL_TOP];
    let mut l: TextLabel = materialize(asset::text_label(
        "",
        content,
        pos,
        [1.0, 1.0, 1.0],
        "center",
    ));
    l.asset_id = id;
    l.font = font;
    l.scale = theme::TEXT_SCALE;
    l
}

fn row_label(
    id: AssetId,
    content: &str,
    rect: [f32; 4],
    font: Option<FontHandle>,
    visible: bool,
) -> TextLabel {
    let pos = [rect[0] + 12.0, rect[1] + 10.0];
    let mut l: TextLabel = materialize(
        asset::text_label("", content, pos, [0.9, 0.9, 0.92], "left").set("visible", visible),
    );
    l.asset_id = id;
    l.font = font;
    l.scale = theme::TEXT_SCALE;
    l
}

fn text_field(id: AssetId, placeholder: &str, font: Option<FontHandle>) -> TextInput {
    let mut t: TextInput = materialize(
        asset::text_input("", placeholder)
            .set("background", [0.14, 0.15, 0.20, 1.0])
            .set("max_len", 48u32)
            .set("visible", false),
    );
    t.asset_id = id;
    t.font = font;
    t.scale = theme::TEXT_SCALE;
    t
}

#[cfg(test)]
mod tests {
    use super::super::{form_panel, panel, preview, template_panel, templates, view};
    use super::*;

    // The editor draws its own chrome, so injection turns the world's title bar
    // off in place -- keeping the rest of the authored window untouched. Where
    // the title bar is the window's only drag handle and close button, the
    // authored value stands instead.
    #[test]
    fn injection_overrides_an_authored_title_bar_only_where_the_chrome_survives() {
        let mut world = World::new();
        world.add_component(Window {
            title: "My Game".into(),
            width: 1600,
            title_bar: true,
            ..Default::default()
        });
        editor_hud(&mut world);

        let w = world.query::<Window>().next().unwrap();
        assert_eq!(w.title_bar, !HIDES_TITLE_BAR);
        // Only the chrome is the editor's business; size and title stay authored.
        assert_eq!(w.title, "My Game");
        assert_eq!(w.width, 1600);
        assert_eq!(world.query::<Window>().count(), 1);
    }

    // A world with no Window of its own would render through Window::default(),
    // whose title_bar is on -- so injection has to materialize one to override.
    // With nothing to override there is no reason to synthesize one.
    #[test]
    fn injection_materializes_a_window_to_override_only_where_the_chrome_survives() {
        let mut world = World::new();
        editor_hud(&mut world);

        let windows: Vec<&Window> = world.query::<Window>().collect();
        if HIDES_TITLE_BAR {
            assert_eq!(windows.len(), 1);
            assert!(!windows[0].title_bar);
        } else {
            assert!(windows.is_empty());
        }
    }

    // A `borderless` / `fullscreen` world would size the editor window to the
    // whole display and swallow the OS cursor past the panels, so injection pins
    // the editor session to windowed on every platform -- while leaving the rest
    // of the authored window (size, title) for `cn run` / `cn export` to honor.
    #[test]
    fn injection_pins_a_non_windowed_mode_to_windowed() {
        for mode in [WindowMode::Borderless, WindowMode::Fullscreen] {
            let mut world = World::new();
            world.add_component(Window {
                title: "My Game".into(),
                width: 1600,
                mode,
                ..Default::default()
            });
            editor_hud(&mut world);

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

    // Injection adds the top-bar controls (visible) plus the Assets, Preview,
    // View, and Templates panels' chrome, row families, and the typed fields (all
    // hidden) -- every one a view-less overlay at a reserved id.
    #[test]
    fn injects_top_bar_and_hidden_panels() {
        let mut world = World::new();
        editor_hud(&mut world);

        // Top-bar buttons are visible.
        for id in [hud::SAVE_BUTTON, hud::VIEW_BUTTON] {
            assert!(
                world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == id)
                    .unwrap()
                    .visible,
                "{id:?} visible"
            );
        }

        // The View button is labelled "View".
        assert_eq!(
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == hud::VIEW_LABEL)
                .unwrap()
                .content,
            "View"
        );

        // Panel chrome + a representative row from each family start hidden, as do
        // the whole Preview, View, and Templates panels (the tick shows them on
        // demand).
        for id in [
            panel::PANEL_BG,
            panel::PLUS_BG,
            panel::PICKER_BG,
            panel::MENU_BG,
            panel::row_bg(0),
            panel::picker_row_bg(0),
            preview::PANEL_BG,
            preview::ROW_BG,
            preview::CHECK_BOX,
            view::PANEL_BG,
            view::row_bg(0),
            view::check_box(0),
            templates::PANEL_BG,
            templates::row_bg(0),
            template_panel::PANEL_BG,
            template_panel::APPLY_BG,
            template_panel::row_bg(0),
        ] {
            assert!(
                !world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == id)
                    .unwrap()
                    .visible,
                "{id:?} starts hidden"
            );
        }

        // Draw order is insertion order: the top bar goes in AFTER the panels so
        // a dragged panel slides behind it (matching the hook's hit-test order).
        let sprites: Vec<AssetId> = world.query::<Sprite>().map(|s| s.asset_id).collect();
        let pos = |id: AssetId| sprites.iter().position(|&x| x == id).unwrap();
        assert!(pos(panel::PANEL_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(preview::PANEL_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(view::PANEL_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(templates::PANEL_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(template_panel::PANEL_BG) < pos(hud::SAVE_BUTTON));

        // Both typed fields exist, hidden, and reference the reused font.
        let fields: Vec<AssetId> = world.query::<TextInput>().map(|t| t.asset_id).collect();
        assert!(fields.contains(&panel::SEARCH_INPUT));
        assert!(fields.contains(&form_panel::NAME_INPUT));
        assert!(world.query::<TextInput>().all(|t| !t.visible));

        // View-less: window space, never overlay-scaled.
        assert!(world.query::<Sprite>().all(|s| s.screen.is_none()));
        assert!(world.query::<TextLabel>().all(|l| l.screen.is_none()));
        assert!(world.query::<TextInput>().all(|t| t.screen.is_none()));
    }

    // The button + field text draws with the engine HUD face, baked into the
    // world's font table here rather than at world start -- NOT the first
    // label in the world, whose font may be a user world's oversized display
    // face. The DebugHud is dropped and its default turned off, since the
    // editor takes over F1.
    #[test]
    fn bakes_the_engine_hud_face_and_takes_over_the_debug_hud() {
        let mut world = World::new();
        world.add_component(TextLabel {
            asset_id: AssetId(0),
            font: Some(FontHandle(99)),
            ..Default::default()
        });
        world.add_component(DebugHud::default());
        editor_hud(&mut world);

        // One face appended, and it is the one the panels use.
        let fonts = world
            .resource::<crate::resource::FontTable>()
            .expect("the face was baked");
        assert_eq!(fonts.len(), 1);
        let baked = FontHandle(0);
        let save = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == hud::SAVE_LABEL)
            .unwrap();
        assert_eq!(save.font, Some(baked));
        let field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap();
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
            sky: false,
            ..Default::default()
        });
        editor_hud(&mut world);

        assert_eq!(world.query::<EngineDefaults>().count(), 1);
        let defaults = world.query::<EngineDefaults>().next().unwrap();
        assert!(!defaults.debug_hud);
        assert!(!defaults.sky, "the world's own opt-out stands");
    }

    // The templates-spec-driven constructors materialize the same components the
    // hand-written struct literals used to, so routing them through templates did
    // not shift any field. Pins the spec builders against drift.
    #[test]
    fn constructors_materialize_expected_components() {
        use crate::components::TextAlign;

        let s = button_sprite(
            AssetId(1),
            [10.0, 20.0, 100.0, 40.0],
            [1.0, 0.0, 0.0, 1.0],
            true,
        );
        assert_eq!((s.x, s.y, s.width, s.height), (10.0, 20.0, 100.0, 40.0));
        assert_eq!(s.tint, [1.0, 0.0, 0.0, 1.0]);
        assert!(s.visible);
        assert_eq!(s.asset_id, AssetId(1));

        let t = text_field(AssetId(2), "name", Some(FontHandle(9)));
        assert_eq!(t.placeholder, "name");
        assert_eq!(t.background, [0.14, 0.15, 0.20, 1.0]);
        assert_eq!(t.max_len, 48);
        assert!(!t.visible);
        assert_eq!(t.font, Some(FontHandle(9)));

        let c = centered_label(AssetId(3), "SAVE", [0.0, 0.0, 88.0, 88.0], None);
        assert_eq!(c.content, "SAVE");
        assert_eq!(c.align, TextAlign::Center);
        assert_eq!(c.x, 44.0, "centered on the button width");
        assert_eq!(c.color, [1.0, 1.0, 1.0]);
        assert!(c.visible);

        let r = row_label(AssetId(4), "row", [0.0, 0.0, 100.0, 40.0], None, false);
        assert_eq!(r.align, TextAlign::Left);
        assert_eq!(r.color, [0.9, 0.9, 0.92]);
        assert!(!r.visible);
    }
}

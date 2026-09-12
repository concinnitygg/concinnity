// src/editor/hook/tests/panels_tests.rs
//
// Contract tests over the panel registry. Each floating panel supplies only its
// own `Panel` impl (`hook/panels.rs`) and the shared machinery drives the rest,
// so the invariants that machinery relies on -- a footprint inside its own max,
// an on-screen origin, a toggle that round-trips, a `hide` that blanks exactly
// what the panel declared for injection, presses and wheels that fall through
// outside the footprint, and scrolling that stays in bounds at either end --
// are asserted here once for every registered panel rather than per panel.
//
// Beside them are the drives of the two panels that are only lists of the
// others: the View panel's rows toggling what shows, and the Templates panel's
// pick opening a detail panel whose apply adds the template's entries.

use concinnity_core::components::{FrameInput, Sprite, TextInput, TextLabel};
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;
use concinnity_host::thread::asset_id::AssetId;

use super::fixtures::{hook, set_input, world_with_input};
use crate::debug_hook::DebugHook;
use crate::editor::hook::EditorHook;
use crate::editor::hud::{self, HudAction};
use crate::editor::panels::registry::{self, Panel, PanelKey};
use crate::editor::panels::template_panel::{self, TemplateAction};
use crate::editor::panels::{list_panel, template, view};
use crate::editor::{inject, widget};
use crate::test_support::isolate_state_dir;

fn empty_hook() -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), Vec::new())
}

// A world carrying every panel's injected elements, which is what the panels'
// `draw` / `hide` mutate.
fn injected_world() -> World {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    world
}

// Open every registered panel, including the ones that have no View row: the
// edit form (gated on a picked type plus the Assets panel), the View panel
// itself, the template detail (gated on an open template), and the command
// palette (opened by its shortcut).
fn open_every_panel(h: &mut EditorHook, world: &mut World) {
    for p in registry::view_toggles() {
        if !p.is_open(h) {
            p.toggle(h, world);
        }
    }
    h.selected_type = Some("Sprite".to_string());
    h.view_open = true;
    h.open_template = Some(0);
    h.palette_open = true;
}

// Every declared element of `p`, forced visible so a `hide` that misses one is
// visible as a leftover rather than passing on an already-blank world.
fn show_every_element(p: &dyn Panel, world: &mut World) {
    for id in p.sprite_ids() {
        widget::set_sprite_visible(world, id, true);
    }
    for id in p.label_ids() {
        widget::set_label_visible(world, id, true);
    }
    for (id, _) in p.field_ids() {
        widget::show_field(world, id, [0.0, 0.0, 10.0, 10.0], false);
    }
}

fn visible_sprites(p: &dyn Panel, world: &World) -> Vec<AssetId> {
    let ids = p.sprite_ids();
    world
        .query::<Sprite>()
        .filter(|s| s.visible && ids.contains(&s.asset_id))
        .map(|s| s.asset_id)
        .collect()
}

// A panel blanks every element it declared for injection. The declared lists are
// what `inject::editor_hud` materializes, so an id a panel lists but never blanks
// would survive the F1 hidden pass (and a toggle-off) as a stranded element.
#[test]
fn hide_blanks_every_declared_element() {
    let mut world = injected_world();
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        show_every_element(p, &mut world);
        p.hide(&mut world);

        for id in p.sprite_ids() {
            let s = world
                .query::<Sprite>()
                .find(|s| s.asset_id == id)
                .unwrap_or_else(|| panic!("{key:?} declares sprite {id:?} but injection has none"));
            assert!(!s.visible, "{key:?} left sprite {id:?} visible after hide");
        }
        for id in p.label_ids() {
            let l = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .unwrap_or_else(|| panic!("{key:?} declares label {id:?} but injection has none"));
            assert!(!l.visible, "{key:?} left label {id:?} visible after hide");
        }
        for (id, _) in p.field_ids() {
            let t = world
                .query::<TextInput>()
                .find(|t| t.asset_id == id)
                .unwrap_or_else(|| panic!("{key:?} declares field {id:?} but injection has none"));
            assert!(!t.visible, "{key:?} left field {id:?} visible after hide");
        }
    }
}

// A panel's default footprint is also its minimum, so it can never start out
// larger than the maximum the resize drag will clamp it to.
#[test]
fn default_size_fits_within_max_size() {
    let h = empty_hook();
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        let s = p.size(&h);
        let max = p.max_size(&h);
        assert!(
            s[0] > 0.0 && s[1] > 0.0,
            "{key:?} has a degenerate default size {s:?}"
        );
        assert!(
            s[0] <= max[0] && s[1] <= max[1],
            "{key:?} default size {s:?} exceeds its max {max:?}"
        );
    }
}

// The stored resize override is clamped into `[default, max]` on read, so a
// stale override from a session where the panel held more content can neither
// shrink it below its content nor grow it past its element pool.
#[test]
fn effective_size_clamps_a_stored_override() {
    for key in PanelKey::ALL {
        let mut h = empty_hook();
        let d = h.default_size(key);
        let max = registry::panel(key).max_size(&h);

        h.sizes[key.index()] = Some([1.0, 1.0]);
        assert_eq!(
            h.effective_size(key),
            d,
            "{key:?} let an override shrink it below its default"
        );

        h.sizes[key.index()] = Some([1.0e6, 1.0e6]);
        let grown = h.effective_size(key);
        assert!(
            grown[0] >= d[0] && grown[1] >= d[1],
            "{key:?} grew below its default"
        );
        for axis in 0..2 {
            if max[axis].is_finite() {
                assert_eq!(
                    grown[axis],
                    max[axis].max(d[axis]),
                    "{key:?} grew past its max on axis {axis}"
                );
            }
        }
    }
}

// Every panel's resting place is fully on screen and clear of the top bar, at
// any window shape -- the clamp is what keeps a panel reachable after a resize.
#[test]
fn every_panel_origin_stays_on_screen() {
    let h = empty_hook();
    for vp in [[1280.0, 720.0], [800.0, 600.0], [3840.0, 2160.0]] {
        for key in PanelKey::ALL {
            let o = h.origin(key, vp);
            let s = h.effective_size(key);
            assert!(o[0] >= 0.0, "{key:?} sits off the left edge at {vp:?}");
            assert!(
                o[1] >= hud::BAR_H,
                "{key:?} sits under the top bar at {vp:?}"
            );
            if s[0] <= vp[0] {
                assert!(
                    o[0] + s[0] <= vp[0],
                    "{key:?} overhangs the right edge at {vp:?}"
                );
            }
            if s[1] <= vp[1] - hud::BAR_H {
                assert!(
                    o[1] + s[1] <= vp[1],
                    "{key:?} overhangs the bottom edge at {vp:?}"
                );
            }
        }
    }
}

// Every View-panel row round-trips its panel in both directions, from whichever
// state that panel defaults to (the Preview panel starts open, the rest closed).
#[test]
fn every_view_toggle_opens_and_closes_its_panel() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    for p in registry::view_toggles() {
        let key = p.key();
        p.close(&mut h, &mut world);
        assert!(!p.is_open(&h), "{key:?} did not close on its title-bar X");

        p.toggle(&mut h, &mut world);
        assert!(p.is_open(&h), "{key:?} did not open on its View row");

        p.toggle(&mut h, &mut world);
        assert!(!p.is_open(&h), "{key:?} did not close on its View row");
    }
}

// The title-bar X shuts every panel, including the three with no View row. Each
// is re-opened first because the compound gates chain: closing the Assets panel
// also takes the edit form with it, and closing Templates the detail panel.
#[test]
fn close_shuts_every_open_panel() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    for key in PanelKey::ALL {
        open_every_panel(&mut h, &mut world);
        let p = registry::panel(key);
        assert!(p.is_open(&h), "{key:?} did not open for the close sweep");
        p.close(&mut h, &mut world);
        assert!(!p.is_open(&h), "{key:?} stayed open after close");
    }
}

// A press or a wheel clear of a panel's footprint is not the panel's: it falls
// through so the panel behind it (or the 3D viewport) gets the event.
#[test]
fn a_press_clear_of_the_footprint_falls_through() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    open_every_panel(&mut h, &mut world);
    let vp = [1280.0, 720.0];
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        let o = h.origin(key, vp);
        let s = h.effective_size(key);
        let (mx, my) = (o[0] + s[0] + 40.0, o[1] + s[1] + 40.0);
        assert!(
            !p.press(&mut h, &mut world, mx, my, o),
            "{key:?} claimed a press outside its footprint"
        );
        assert!(
            !p.wheel_over(&h, &world, mx, my, o),
            "{key:?} claimed a wheel outside its footprint"
        );
    }
}

// Scrolling past either end of a panel's list is a no-op rather than an
// underflow: the offsets are unsigned, so an unclamped step would panic here.
#[test]
fn scrolling_past_either_end_stays_in_bounds() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    open_every_panel(&mut h, &mut world);
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        for _ in 0..32 {
            p.scroll(&mut h, &mut world, 1.0);
        }
        for _ in 0..64 {
            p.scroll(&mut h, &mut world, -1.0);
        }
        for _ in 0..64 {
            p.scroll(&mut h, &mut world, 1.0);
        }
    }
}

// A panel shows chrome when drawn and blanks it again when hidden, so the
// per-frame draw and the F1 hidden pass agree on the same element set.
#[test]
fn draw_shows_chrome_that_hide_takes_back() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    open_every_panel(&mut h, &mut world);
    let vp = [1280.0, 720.0];
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        let o = h.origin(key, vp);
        p.hide(&mut world);
        p.draw(&h, &mut world, o, [o[0] + 4.0, o[1] + 4.0]);
        assert!(
            !visible_sprites(p, &world).is_empty(),
            "{key:?} drew no chrome while open"
        );
        p.hide(&mut world);
        assert!(
            visible_sprites(p, &world).is_empty(),
            "{key:?} left chrome behind after hide"
        );
    }
}

// A panel's floating overlay ids are drawn above its body, so they must come
// from the same declared element set injection materialized.
#[test]
fn overlay_ids_are_declared_elements() {
    let mut h = empty_hook();
    h.field_dropdown = None;
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        let declared = p.sprite_ids();
        let labels = p.label_ids();
        for id in p.overlay_ids(&h) {
            assert!(
                declared.contains(&id) || labels.contains(&id),
                "{key:?} overlay id {id:?} is not among its injected elements"
            );
        }
    }
}

// The top-bar View button toggles the View panel; the View panel's rows toggle
// the Assets, Preview, and Templates panels independently (no mutual exclusion).
#[test]
fn view_button_and_view_rows_toggle_the_panels() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.apply_top(HudAction::ToggleView, &mut world);
    assert!(h.view_open, "the View button shows the View panel");
    h.apply_top(HudAction::ToggleView, &mut world);
    assert!(!h.view_open, "a second click hides it");
    // Row 0 -> Assets, row 1 -> Preview, row 2 -> Templates.
    h.toggle_view_row(0, &mut world);
    assert!(h.panel_open, "row 0 shows the Assets panel");
    h.toggle_view_row(1, &mut world);
    assert!(
        !h.preview_open,
        "row 1 hides the (default-shown) Preview panel"
    );
    h.toggle_view_row(2, &mut world);
    assert!(h.templates_open, "row 2 shows the Templates panel");
    assert!(
        h.panel_open,
        "Assets stayed shown -- panels are independent"
    );
}

// Picking a template opens its detail panel (nothing is added yet); Apply from
// the detail layers the assets and closes it; re-applying is idempotent.
#[test]
fn template_pick_opens_detail_then_apply_adds_idempotently() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    assert_eq!(h.open_template, Some(0), "the detail panel opens on pick");
    assert!(h.entries.is_empty(), "picking adds nothing on its own");

    h.apply_template_detail(TemplateAction::Apply);
    let first = concinnity_cook::authoring::template::TEMPLATES[0]
        .assets()
        .len();
    assert_eq!(h.entries.len(), first, "Apply adds all template entries");
    assert_eq!(h.open_template, None, "Apply closes the detail panel");

    // Re-open and Apply again: no duplicate entries.
    h.open_template_detail(0);
    h.apply_template_detail(TemplateAction::Apply);
    assert_eq!(h.entries.len(), first, "re-apply is idempotent");
}

// The detail panel's grouped rows come from the shared list model, so they
// match what the template would add (one row per asset plus a type header
// each), and the "X" closes the panel without adding anything.
#[test]
fn template_detail_rows_and_close() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    let rows = h.template_rows(0);
    let names = rows.iter().filter(|r| !r.is_header).count();
    assert_eq!(
        names,
        concinnity_cook::authoring::template::TEMPLATES[0]
            .assets()
            .len(),
        "one name row per template asset"
    );
    assert!(
        rows.iter().any(|r| r.is_header),
        "grouped under type headers"
    );
    h.apply_template_detail(TemplateAction::Close);
    assert_eq!(h.open_template, None);
    assert!(h.entries.is_empty(), "closing adds nothing");
}

// Clicking the Preview panel's capture row hands the cursor to the world
// (clicking again takes it back); the fly row below toggles the fly camera,
// and the two never hold the cursor together.
#[test]
fn preview_rows_toggle_play_mode_and_fly() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let o = h.origin(PanelKey::Preview, vp);
    let row_mid = |i: usize| {
        let r = list_panel::row_rect(o, 200.0, i);
        [r[0] + 10.0, r[1] + r[3] * 0.5]
    };
    let click = |h: &mut EditorHook, world: &mut World, pos: [f32; 2]| {
        set_input(
            world,
            FrameInput {
                viewport: vp,
                mouse_x: pos[0],
                mouse_y: pos[1],
                left_click: true,
                left_button_down: true,
                ..Default::default()
            },
        );
        h.tick(world);
    };
    let mut world = world_with_input(FrameInput::default());

    click(&mut h, &mut world, row_mid(0));
    assert!(h.sim.playing(), "the checkbox click enters play mode");
    click(&mut h, &mut world, row_mid(0));
    assert!(!h.sim.playing(), "a second click leaves it");

    click(&mut h, &mut world, row_mid(1));
    assert!(h.fly, "the fly row starts the fly camera");
    assert!(!h.sim.playing());
    click(&mut h, &mut world, row_mid(0));
    assert!(
        h.sim.playing() && !h.fly,
        "entering play mode ends the fly camera"
    );
}

// End-to-end through `tick` against a fully injected HUD: the top-bar View
// button opens the View panel, and clicking its "Templates" row opens the
// Templates panel (the same click path a real session drives).
#[test]
fn tick_view_button_opens_view_then_a_row_opens_templates() {
    let vis = |w: &World, id: asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .map(|s| s.visible)
            .unwrap_or(false)
    };
    let rect = |w: &World, id: asset_id::AssetId| {
        let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
        [s.x, s.y, s.width, s.height]
    };
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());

    // Frame 1: no interaction. View + Templates start hidden.
    world.add_component(FrameInput {
        viewport: vp,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(!vis(&world, view::PANEL_BG) && !vis(&world, template::PANEL_BG));

    // Frame 2: click the top-bar View button -> the View panel opens.
    let view_btn = hud::layout(vp[0]).view;
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: view_btn[0] + view_btn[2] * 0.5,
            mouse_y: view_btn[1] + view_btn[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(
        h.view_open && vis(&world, view::PANEL_BG),
        "View panel opened"
    );
    // Its "Templates" row (index 2) is laid out; grab its rect to click it.
    let row = rect(&world, view::row_bg(2));

    // Frame 3: click that row -> the Templates panel opens.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: row[0] + row[2] * 0.5,
            mouse_y: row[1] + row[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.templates_open, "the Templates row toggled the panel on");
    assert!(vis(&world, template::PANEL_BG), "Templates panel shown");
}

// Picking a template row spawns the detail panel (title "Template <name>",
// hidden until then); its Apply button layers the template's assets and closes
// the detail. Drives the whole flow through `tick` end to end.
#[test]
fn tick_picking_a_template_spawns_the_detail_panel_then_apply_adds() {
    let vis = |w: &World, id: asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .map(|s| s.visible)
            .unwrap_or(false)
    };
    let rect = |w: &World, id: asset_id::AssetId| {
        let s = w.query::<Sprite>().find(|s| s.asset_id == id).unwrap();
        [s.x, s.y, s.width, s.height]
    };
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());
    // Start with the Templates list already open.
    h.templates_open = true;
    world.add_component(FrameInput {
        viewport: vp,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(
        vis(&world, template::PANEL_BG) && !vis(&world, template_panel::PANEL_BG),
        "Templates list shown; detail panel still hidden"
    );

    // Click the first template row -> the detail panel spawns.
    let row = rect(&world, template::row_bg(0));
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: row[0] + row[2] * 0.5,
            mouse_y: row[1] + row[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.open_template, Some(0), "the detail panel opened on pick");
    assert!(vis(&world, template_panel::PANEL_BG), "detail panel shown");
    let title = world
        .query::<TextLabel>()
        .find(|l| l.asset_id == template_panel::TITLE_LABEL)
        .unwrap();
    assert!(
        title.content.starts_with("Template "),
        "title bar reads 'Template <name>': {}",
        title.content
    );
    assert!(h.entries.is_empty(), "picking adds nothing yet");

    // Click the detail's Apply button -> the template's assets are added and
    // the detail closes.
    let apply = template_panel::apply_rect(
        h.origin(PanelKey::TemplateDetail, vp),
        h.effective_size(PanelKey::TemplateDetail)[0],
    );
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: apply[0] + apply[2] * 0.5,
            mouse_y: apply[1] + apply[3] * 0.5,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.open_template, None, "Apply closed the detail panel");
    assert!(
        !vis(&world, template_panel::PANEL_BG),
        "detail panel hidden"
    );
    assert_eq!(
        h.entries.len(),
        concinnity_cook::authoring::template::TEMPLATES[0]
            .assets()
            .len(),
        "Apply layered the template's assets"
    );
}

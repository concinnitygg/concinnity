//! Contract tests over the panel registry. Each floating panel supplies only its
//! own `Panel` impl (`hook/panels.rs`) and the shared machinery drives the rest,
//! so the invariants that machinery relies on -- a footprint inside its own max,
//! an on-screen origin, a toggle that round-trips, a `hide` that blanks exactly
//! what the panel declared for injection, presses and wheels that fall through
//! outside the footprint, and scrolling that stays in bounds at either end --
//! are asserted here once for every registered panel rather than per panel.
//!
//! Beside them are the drives of the two panels that are only lists of the
//! others: the View panel's rows toggling what shows, and the Templates panel's
//! pick opening a detail panel whose apply adds the template's entries.

use concinnity_engine::components::{
    FrameInput, Layout, ShaderStage, Sprite, TextField, TextLabel, UiEvent, UiState, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::fixtures::{built_shown, hook, send_event, set_input};
use crate::editor::hook::EditorHook;
use crate::editor::hook::edit::shaders_state::SourceState;
use crate::editor::hud::{self, HudAction};
use crate::editor::panels::registry::{self, Panel, PanelKey};
use crate::editor::panels::shader::source::SourceKey;
use crate::editor::panels::view;
use crate::editor::ui::BuiltUi;
use crate::editor::{inject, widget};
use crate::frame_hook::FrameHook;
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
    h.form.selected_type = Some("Sprite".to_string());
    h.open[PanelKey::View] = true;
    h.templates.detail = Some(0);
    h.open[PanelKey::Palette] = true;
    h.shaders.source = Some(SourceState::new(
        SourceKey::shader("lit", ShaderStage::Fragment),
        "/cn-none/lit.hlsl".to_string(),
        "float4 shade(VertexOut v, GpuObjectData od) { return 1.0; }".to_string(),
    ));
}

// Every declared element of `p`, forced visible so a `hide` that misses one is
// visible as a leftover rather than passing on an already-blank world.
fn show_every_element(p: &dyn Panel, world: &mut World) {
    let ids = p.ids();
    for &id in &ids.sprites {
        widget::set_sprite_visible(world, id, true);
    }
    for &id in ids.labels.iter().chain(&ids.code_labels) {
        widget::set_label_visible(world, id, true);
    }
    for id in ids.field_ids() {
        widget::show_field(world, id, [0.0, 0.0, 10.0, 10.0], false);
    }
}

fn visible_sprites(p: &dyn Panel, world: &World) -> Vec<AssetId> {
    let ids = &p.ids().sprites;
    world
        .join2::<Sprite, concinnity_engine::components::Identity>()
        .filter(|(_, s, identity)| s.visible && ids.contains(&identity.id()))
        .map(|(_, _, identity)| identity.id())
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

        let ids = p.ids();
        for &id in &ids.sprites {
            let s = world
                .get_by_id::<Sprite>(id)
                .unwrap_or_else(|| panic!("{key:?} declares sprite {id:?} but injection has none"));
            assert!(!s.visible, "{key:?} left sprite {id:?} visible after hide");
        }
        for &id in ids.labels.iter().chain(&ids.code_labels) {
            let l = world
                .get_by_id::<TextLabel>(id)
                .unwrap_or_else(|| panic!("{key:?} declares label {id:?} but injection has none"));
            assert!(!l.visible, "{key:?} left label {id:?} visible after hide");
        }
        for id in ids.field_ids() {
            let t = world
                .get_by_id::<TextField>(id)
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

        h.sizes[key] = Some([1.0, 1.0]);
        assert_eq!(
            h.effective_size(key),
            d,
            "{key:?} let an override shrink it below its default"
        );

        h.sizes[key] = Some([1.0e6, 1.0e6]);
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

// The title-bar X shuts every panel, including the ones with no View row. Each
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

// A hand-drawn panel shows chrome when drawn and blanks it again when hidden,
// so the per-frame draw and the F1 hidden pass agree on the same element set.
#[test]
fn draw_shows_chrome_that_hide_takes_back() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = empty_hook();
    open_every_panel(&mut h, &mut world);
    let vp = [1280.0, 720.0];
    for key in PanelKey::ALL
        .into_iter()
        .filter(|&k| !registry::panel(k).is_built())
    {
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
    h.form.field_dropdown = None;
    for key in PanelKey::ALL {
        let p = registry::panel(key);
        for id in p.overlay_ids(&h) {
            assert!(
                p.ids().all().any(|declared| declared == id),
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
    assert!(
        h.open[PanelKey::View],
        "the View button shows the View panel"
    );
    h.apply_top(HudAction::ToggleView, &mut world);
    assert!(!h.open[PanelKey::View], "a second click hides it");
    // Row 0 -> Assets, row 1 -> Preview, row 2 -> Templates.
    h.toggle_view_row(0, &mut world);
    assert!(h.open[PanelKey::Assets], "row 0 shows the Assets panel");
    h.toggle_view_row(1, &mut world);
    assert!(
        !h.open[PanelKey::Preview],
        "row 1 hides the (default-shown) Preview panel"
    );
    h.toggle_view_row(2, &mut world);
    assert!(
        h.open[PanelKey::Templates],
        "row 2 shows the Templates panel"
    );
    assert!(
        h.open[PanelKey::Assets],
        "Assets stayed shown -- panels are independent"
    );
}

// Picking a template opens its detail panel (nothing is added yet); Apply from
// the detail layers the assets and closes it; re-applying is idempotent.
#[test]
fn template_pick_opens_detail_then_apply_adds_idempotently() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    assert_eq!(
        h.templates.detail,
        Some(0),
        "the detail panel opens on pick"
    );
    assert!(h.entries.is_empty(), "picking adds nothing on its own");

    h.apply_open_template();
    let first = concinnity_cook::authoring::template::TEMPLATES[0]
        .assets()
        .len();
    assert_eq!(h.entries.len(), first, "Apply adds all template entries");
    assert_eq!(h.templates.detail, None, "Apply closes the detail panel");

    // Re-open and Apply again: no duplicate entries.
    h.open_template_detail(0);
    h.apply_open_template();
    assert_eq!(h.entries.len(), first, "re-apply is idempotent");
}

// The detail panel's grouped rows come from the shared list model, so they
// match what the template would add (one row per asset plus a type header
// each), and the "X" closes the panel without adding anything.
#[test]
fn template_detail_rows_and_close() {
    let mut h = hook(Vec::new());
    h.open_template_detail(0);
    let rows = EditorHook::template_rows(0);
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
    h.close_template_detail();
    assert_eq!(h.templates.detail, None);
    assert!(h.entries.is_empty(), "closing adds nothing");
}

// The Preview panel's play checkbox hands the cursor to the world (flipping
// it again takes it back); the fly checkbox toggles the fly camera, and the
// two never hold the cursor together.
#[test]
fn preview_rows_toggle_play_mode_and_fly() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    let mut send = |h: &mut EditorHook, name: &str| {
        world.events_mut::<UiEvent>().send(UiEvent {
            name: name.into(),
            element: None,
            value: None,
        });
        h.read_ui_events(&mut world);
    };
    send(&mut h, "editor.preview.play");
    assert!(h.sim.playing(), "the checkbox enters play mode");
    send(&mut h, "editor.preview.play");
    assert!(!h.sim.playing(), "a second flip leaves it");

    send(&mut h, "editor.preview.fly");
    assert!(h.fly, "the fly checkbox starts the fly camera");
    assert!(!h.sim.playing());
    send(&mut h, "editor.preview.play");
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
    let view_shown = |w: &World| {
        let built = w
            .resource::<BuiltUi>()
            .unwrap()
            .panel(PanelKey::View)
            .unwrap();
        w.get_by_id::<Layout>(built.root).unwrap().visible
    };
    assert!(!view_shown(&world) && !built_shown(&world, PanelKey::Templates));

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
        h.open[PanelKey::View] && view_shown(&world),
        "View panel opened"
    );

    // Frame 3: its "Templates" row (index 2) is flipped -> the Templates panel
    // opens.
    world.events_mut::<UiEvent>().send(view::row_event(2));
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(
        h.open[PanelKey::Templates],
        "the Templates row toggled the panel on"
    );
    assert!(
        built_shown(&world, PanelKey::Templates),
        "Templates panel shown"
    );
}

// Picking a template row opens the detail panel (title "Template <name>",
// hidden until then); its Apply button layers the template's assets and closes
// the detail. Drives the whole flow through `tick` end to end.
#[test]
fn tick_picking_a_template_spawns_the_detail_panel_then_apply_adds() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());
    // Start with the Templates list already open.
    h.open[PanelKey::Templates] = true;
    let idle = FrameInput {
        viewport: vp,
        ..Default::default()
    };
    world.add_component(idle.clone());
    h.tick(&mut world);
    assert!(
        built_shown(&world, PanelKey::Templates) && !built_shown(&world, PanelKey::TemplateDetail),
        "Templates list shown; detail panel still hidden"
    );

    // Selecting the first template row -> the detail panel opens.
    send_event(&mut world, "editor.templates.pick", Some(UiValue::Int(0)));
    set_input(&mut world, idle.clone());
    h.tick(&mut world);
    assert_eq!(
        h.templates.detail,
        Some(0),
        "the detail panel opened on pick"
    );
    assert!(
        built_shown(&world, PanelKey::TemplateDetail),
        "detail panel shown"
    );
    let title = world
        .resource::<UiState>()
        .unwrap()
        .get("editor.template.title");
    assert!(
        matches!(title, Some(UiValue::Text(t)) if t.starts_with("Template ")),
        "title bar reads 'Template <name>': {title:?}"
    );
    assert!(h.entries.is_empty(), "picking adds nothing yet");

    // The detail's Apply button -> the template's assets are added and the
    // detail closes.
    send_event(&mut world, "editor.template.apply", None);
    set_input(&mut world, idle);
    h.tick(&mut world);
    assert_eq!(h.templates.detail, None, "Apply closed the detail panel");
    assert!(
        !built_shown(&world, PanelKey::TemplateDetail),
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

// A built panel follows the hook's model into a fresh world, keeps the press
// that lands on it from the panels and world behind, and closes on the event
// its close button sends.
#[test]
fn a_built_panel_takes_its_presses_and_closes_on_its_event() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let vp = [1280.0, 720.0];
    let mut h = hook(Vec::new());
    h.open[PanelKey::View] = true;
    world.add_component(FrameInput {
        viewport: vp,
        ..Default::default()
    });
    h.tick(&mut world);
    let built = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::View)
        .unwrap()
        .clone();
    let layout = world.get_by_id::<Layout>(built.root).unwrap();
    assert!(layout.visible);
    assert_eq!(
        layout.offset,
        h.origin(PanelKey::View, vp),
        "placed where the hook keeps it"
    );
    // The engine lays it out; until then nothing claims a press for it.
    let o = layout.offset;
    world.get_mut_by_id::<Layout>(built.root).unwrap().rect = Some([o[0], o[1], 200.0, 300.0]);
    assert!(h.try_panel_press(PanelKey::View, o[0] + 20.0, o[1] + 60.0, vp, &mut world));
    assert!(!h.editor_took_press, "the engine handles the press itself");
    assert_eq!(h.panel_order.last(), Some(&PanelKey::View));

    world.events_mut::<UiEvent>().send(UiEvent {
        name: crate::editor::ui::CLOSE_EVENT.into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(PanelKey::View.index()).unwrap())),
    });
    h.tick(&mut world);
    assert!(!h.open[PanelKey::View]);
    assert!(!world.get_by_id::<Layout>(built.root).unwrap().visible);
}

// A panel the engine raised on a press comes to the front of the hook's
// focus stack too.
#[test]
fn a_built_panel_the_engine_raised_comes_forward() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.open[PanelKey::View] = true;
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    h.focus_panel(PanelKey::Preview);
    h.tick(&mut world);
    assert_ne!(h.panel_order.last(), Some(&PanelKey::View));
    let root = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::View)
        .unwrap()
        .root;
    world.get_mut_by_id::<Layout>(root).unwrap().rank += 100;
    h.tick(&mut world);
    assert_eq!(h.panel_order.last(), Some(&PanelKey::View));
}

// A built panel dragged under the top bar, or left off a shrunk window, is put
// back on screen below the bar.
#[test]
fn a_built_panel_stays_on_screen_below_the_top_bar() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.open[PanelKey::View] = true;
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    let root = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::View)
        .unwrap()
        .root;
    // The engine laid it out where a drag left it, under the bar.
    let layout = world.get_mut_by_id::<Layout>(root).unwrap();
    layout.offset = [40.0, 0.0];
    layout.rect = Some([40.0, 0.0, 200.0, 300.0]);
    h.tick(&mut world);
    let offset = world.get_by_id::<Layout>(root).unwrap().offset;
    assert_eq!(offset[0], 40.0);
    assert!(offset[1] >= hud::BAR_H, "pushed below the bar: {offset:?}");
}

// A built panel whose content changes size is laid out at that size.
#[test]
fn a_built_panel_takes_the_size_its_content_asks_for() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.open[PanelKey::Templates] = true;
    h.open_template_detail(0);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    let root = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::TemplateDetail)
        .unwrap()
        .root;
    let want = h.default_size(PanelKey::TemplateDetail);
    let layout = world.get_by_id::<Layout>(root).unwrap();
    assert_eq!(
        layout.height,
        concinnity_engine::components::LayoutSize::Fixed(want[1])
    );
    assert_eq!(layout.min_size, want);
}

// Events a new world's widgets send reach the hook, whatever it had read in
// the world before.
#[test]
fn a_rebuilt_worlds_events_are_read_from_its_start() {
    let mut h = hook(Vec::new());
    let mut old = World::new();
    inject::editor_hud(&mut old);
    old.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    for _ in 0..3 {
        send_event(&mut old, "editor.unrelated", None);
    }
    h.tick(&mut old);
    h.tick(&mut old);

    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    send_event(&mut world, "editor.preview.axes", None);
    let axes = h.axes_visible;
    h.tick(&mut world);
    assert_ne!(
        h.axes_visible, axes,
        "the first event in the new world landed"
    );
}

// A size the user dragged a built panel to stays inside what its content
// allows.
#[test]
fn a_dragged_size_is_held_to_the_panels_bounds() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.open[PanelKey::Templates] = true;
    h.open_template_detail(0);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    let root = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::TemplateDetail)
        .unwrap()
        .root;
    world.get_mut_by_id::<Layout>(root).unwrap().size = Some([10.0, 10.0]);
    h.tick(&mut world);
    let want = h.default_size(PanelKey::TemplateDetail);
    assert_eq!(world.get_by_id::<Layout>(root).unwrap().size, Some(want));
}

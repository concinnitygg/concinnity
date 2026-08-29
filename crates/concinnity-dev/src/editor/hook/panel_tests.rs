// src/editor/hook/panel_tests.rs
//
// Contract tests over the panel registry. Each floating panel supplies only its
// own `Panel` impl (`hook/panels.rs`) and the shared machinery drives the rest,
// so the invariants that machinery relies on -- a footprint inside its own max,
// an on-screen origin, a toggle that round-trips, a `hide` that blanks exactly
// what the panel declared for injection, presses and wheels that fall through
// outside the footprint, and scrolling that stays in bounds at either end --
// are asserted here once for every registered panel rather than per panel.

use super::*;
use crate::editor::inject;
use crate::editor::registry::Panel;
use crate::test_support::isolate_state_dir;

fn hook() -> EditorHook {
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
        .query::<crate::components::Sprite>()
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
                .query::<crate::components::Sprite>()
                .find(|s| s.asset_id == id)
                .unwrap_or_else(|| panic!("{key:?} declares sprite {id:?} but injection has none"));
            assert!(!s.visible, "{key:?} left sprite {id:?} visible after hide");
        }
        for id in p.label_ids() {
            let l = world
                .query::<crate::components::TextLabel>()
                .find(|l| l.asset_id == id)
                .unwrap_or_else(|| panic!("{key:?} declares label {id:?} but injection has none"));
            assert!(!l.visible, "{key:?} left label {id:?} visible after hide");
        }
        for (id, _) in p.field_ids() {
            let t = world
                .query::<crate::components::TextInput>()
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
    let h = hook();
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
        let mut h = hook();
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
    let h = hook();
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
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = hook();
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
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = hook();
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
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = hook();
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
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = hook();
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
    isolate_state_dir();
    let mut world = injected_world();
    let mut h = hook();
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
    let mut h = hook();
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

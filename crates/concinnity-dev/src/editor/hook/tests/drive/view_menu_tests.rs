//! The Display menu (`hook/drive/view_menu.rs`): the render mode and the
//! viewport flags its picks set, and the close it adopts from the menu.

use concinnity_engine::components::{UiEvent, UiValue};
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;

use crate::editor::view_menu;

fn pick(h: &mut EditorHook, row: view_menu::MenuRow) {
    let i = view_menu::rows().iter().position(|r| *r == row).unwrap();
    let event = UiEvent {
        name: "editor.display.pick".into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(i).unwrap())),
    };
    assert!(h.on_display_menu_event(&event));
}

// Display-menu picks act in place (mode radio, flag toggle), and the menu
// closing itself closes it in the hook.
#[test]
fn display_menu_picks_set_mode_and_flags() {
    let mut h = hook(Vec::new());
    h.display_menu_open = true;

    pick(
        &mut h,
        view_menu::MenuRow::Mode(view_menu::ViewMode::Wireframe),
    );
    assert_eq!(h.view_mode, view_menu::ViewMode::Wireframe);
    assert!(h.display_menu_open, "a mode pick keeps the menu up");

    let fog = view_menu::ShowFlags::FOG;
    pick(&mut h, view_menu::MenuRow::Flag(fog, "Fog"));
    assert!(!h.show_flags.contains(fog));
    pick(&mut h, view_menu::MenuRow::Flag(fog, "Fog"));
    assert!(h.show_flags.contains(fog));

    pick(&mut h, view_menu::MenuRow::Billboards);
    assert!(!h.show_billboards);

    // The menu writes its own close; the hook adopts it.
    let mut world = World::new();
    h.observe_display_menu(&world);
    assert!(!h.display_menu_open);
    h.display_menu_open = true;
    crate::editor::ui::state::set_bool(&mut world, view_menu::OPEN, true);
    h.observe_display_menu(&world);
    assert!(h.display_menu_open, "still open while the menu is");
}

// A world the menu was just built into has never shown it, so its unset open
// state is no close.
#[test]
fn a_fresh_world_keeps_the_menu_open() {
    let mut h = hook(Vec::new());
    h.display_menu_open = true;
    let mut world = World::new();
    crate::editor::install::editor_ui(&mut world);
    h.observe_display_menu(&world);
    assert!(h.display_menu_open);
}

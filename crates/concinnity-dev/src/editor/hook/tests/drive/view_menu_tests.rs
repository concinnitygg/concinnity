// src/editor/hook/tests/drive/view_menu_tests.rs
//
// The Display menu (`hook/drive/view_menu.rs`): the render mode and the
// viewport flags its rows set.

use concinnity_core::components::FrameInput;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;

use crate::editor::view_menu;

// Display-menu rows act in place (mode radio, flag toggle) and a click away
// below the bar dismisses the menu.
#[test]
fn display_menu_rows_set_mode_and_flags() {
    let mut h = hook(Vec::new());
    h.display_menu_open = true;
    let vp = [1280.0_f32, 720.0];
    let click_row = |h: &mut EditorHook, row: view_menu::MenuRow| {
        let i = view_menu::rows().iter().position(|r| *r == row).unwrap();
        let r = view_menu::row_rect(vp[0], i);
        let input = FrameInput {
            mouse_x: r[0] + 2.0,
            mouse_y: r[1] + 2.0,
            ..Default::default()
        };
        assert!(
            h.route_display_menu_click(&input, vp),
            "menu consumes the press"
        );
    };

    click_row(
        &mut h,
        view_menu::MenuRow::Mode(view_menu::ViewMode::Wireframe),
    );
    assert_eq!(h.view_mode, view_menu::ViewMode::Wireframe);
    assert!(h.display_menu_open, "a mode pick keeps the menu up");

    let fog = view_menu::ShowFlags::FOG;
    click_row(&mut h, view_menu::MenuRow::Flag(fog, "Fog"));
    assert!(!h.show_flags.contains(fog));
    click_row(&mut h, view_menu::MenuRow::Flag(fog, "Fog"));
    assert!(h.show_flags.contains(fog));

    click_row(&mut h, view_menu::MenuRow::Billboards);
    assert!(!h.show_billboards);

    // A press away from the menu (below the bar) dismisses and consumes.
    let away = FrameInput {
        mouse_x: 20.0,
        mouse_y: 300.0,
        ..Default::default()
    };
    assert!(h.route_display_menu_click(&away, vp));
    assert!(!h.display_menu_open);
    // Closed: presses route normally again.
    assert!(!h.route_display_menu_click(&away, vp));
}

// src/editor/hook/tests/edit/asset_tree_tests.rs
//
// The Assets panel's tree (`hook/edit/asset_tree.rs`): the rows an origin
// grouping produces and how a search narrows them, when the tree is cooked and
// what restales it, folding a group, and the select-and-edit a row click
// performs.

use concinnity_core::ecs::World;

use crate::editor::hook::tests::fixtures::{
    click_row, entry, expandable_hook, generated_group, hook, seed_tree, set_field,
    world_with_fields,
};
use crate::editor::hook::{EditorHook, FormTarget};

use crate::editor::panels::asset_tree::{self, TreeRow};

use crate::editor::panels::panel::{self, PanelAction};

// The tree lists the world's own lines under `World` and each expansion's
// output under whatever produced it, with the search field narrowing both.
#[test]
fn tree_rows_group_by_origin_and_narrow_by_search() {
    let mut h = hook(vec![entry("lamp", "PointLight"), entry("sign", "Decal")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    seed_tree(
        &mut h,
        vec![generated_group("fox", &[("fox_mat", "Material")])],
    );

    let names = |h: &EditorHook, w: &World| -> Vec<String> {
        h.tree_rows(w)
            .iter()
            .filter_map(|r| match r {
                TreeRow::Asset { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(names(&h, &world), ["lamp", "sign", "fox_mat"]);
    let headers: Vec<String> = h
        .tree_rows(&world)
        .iter()
        .filter_map(|r| match r {
            TreeRow::Header { label, .. } => Some(label.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(headers, [asset_tree::WORLD_GROUP, "fox"]);

    // The search field matches name or type, across every group.
    set_field(&mut world, panel::SEARCH_INPUT, "mat");
    assert_eq!(names(&h, &world), ["fox_mat"]);
    set_field(&mut world, panel::SEARCH_INPUT, "PointLight");
    assert_eq!(names(&h, &world), ["lamp"], "a type match reaches the row");
}

// The model is only cooked when the panel is actually showing: a closed panel
// must not pay for an expansion it never draws.
#[test]
fn the_tree_is_cooked_only_while_the_panel_is_up() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    h.panel_open = false;
    assert!(h.tree_stale);
    h.refresh_tree_if_needed();
    assert!(
        h.tree_stale && h.tree_groups.is_empty(),
        "a hidden panel does not cook"
    );

    h.panel_open = true;
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale);
    assert!(
        !h.tree_groups.is_empty(),
        "the injected companions are grouped"
    );
    assert_eq!(h.tree_status, None);
}

// An edit invalidates the model, and the refresh happens once for a burst of
// edits rather than once per edit.
#[test]
fn an_edit_restales_the_tree() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale);
    h.mark_changed();
    assert!(h.tree_stale, "the expansion follows the entries");
    h.mark_changed();
    h.refresh_tree_if_needed();
    assert!(!h.tree_stale, "one refresh covers the burst");
}

#[test]
fn groups_fold_and_unfold() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    let world = world_with_fields();
    h.refresh_tree_if_needed();
    let folded = h.tree_rows(&world).len();
    assert_eq!(folded, h.tree_groups.len(), "headers only");

    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert!(h.tree_rows(&world).len() > folded, "the group unfolded");
    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert_eq!(h.tree_rows(&world).len(), folded, "and folded again");
}

// A tree row click mirrors a viewport pick: plain replaces the selection and
// opens the clicked entry's edit form; with shift held it toggles membership
// instead.
#[test]
fn tree_row_click_selects_and_opens_the_form() {
    let mut world = world_with_fields();
    let mut h = hook(vec![entry("box", "Sprite"), entry("cam", "Camera3D")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "box", &mut world);
    assert_eq!(h.selection.active(), Some("box"));
    assert!(h.panel_open, "the assets UI comes up around the form");
    assert_eq!(
        h.form_target,
        FormTarget::Entry(0),
        "the form targets the clicked entry"
    );

    h.shift_held = true;
    click_row(&mut h, "cam", &mut world);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box", "cam"],
        "a shift click adds"
    );
    click_row(&mut h, "cam", &mut world);
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
        ["box"],
        "a second shift click removes"
    );
}

//! The Assets panel's tree (`hook/edit/asset_tree.rs`): the rows an origin
//! grouping produces and how a search narrows them, when the tree is cooked and
//! what restales it, folding a group, and the select-and-edit a row click
//! performs.

use concinnity_cook::authoring::registry::RegisteredType;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    active, click_row, entry, entry_target, expandable_hook, generated_group, hook, row_of,
    seed_tree, selected, world_with_fields,
};

use crate::editor::panels::asset_tree::{self, TreeRow};

use crate::editor::panels::assets_panel::PanelAction;
use crate::editor::panels::registry::PanelKey;

// The tree lists the world's own lines under `World` and each expansion's
// output under whatever produced it, with the search field narrowing both.
#[test]
fn tree_rows_group_by_origin_and_narrow_by_search() {
    let mut h = hook(vec![entry("lamp", "PointLight"), entry("sign", "Decal")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(
        &mut h,
        vec![generated_group(
            "fox",
            &[("fox_mat", RegisteredType::Material)],
        )],
    );

    let names = |h: &EditorHook| -> Vec<String> {
        h.tree_rows()
            .iter()
            .filter_map(|r| match r {
                TreeRow::Asset { name, .. } => Some(name.clone()),
                TreeRow::Header { .. } => None,
            })
            .collect()
    };
    assert_eq!(names(&h), ["lamp", "sign", "fox_mat"]);
    let headers: Vec<String> = h
        .tree_rows()
        .iter()
        .filter_map(|r| match r {
            TreeRow::Header { label, .. } => Some(label.clone()),
            TreeRow::Asset { .. } => None,
        })
        .collect();
    assert_eq!(headers, [asset_tree::WORLD_GROUP, "fox"]);

    // The search field matches name or type, across every group.
    h.assets.filter = "mat".into();
    assert_eq!(names(&h), ["fox_mat"]);
    h.assets.filter = "PointLight".into();
    assert_eq!(names(&h), ["lamp"], "a type match reaches the row");
}

// The model is only cooked when the panel is actually showing: a closed panel
// must not pay for an expansion it never draws.
#[test]
fn the_tree_is_cooked_only_while_the_panel_is_up() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    h.open[PanelKey::Assets] = false;
    assert!(h.assets.stale);
    h.refresh_tree_if_needed();
    assert!(
        h.assets.stale && h.assets.groups.is_empty(),
        "a hidden panel does not cook"
    );

    h.open[PanelKey::Assets] = true;
    h.refresh_tree_if_needed();
    assert!(!h.assets.stale);
    assert!(
        !h.assets.groups.is_empty(),
        "the injected companions are grouped"
    );
    assert_eq!(h.assets.status, None);
}

// An edit invalidates the model, and the refresh happens once for a burst of
// edits rather than once per edit.
#[test]
fn an_edit_restales_the_tree() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    h.refresh_tree_if_needed();
    assert!(!h.assets.stale);
    h.mark_changed();
    assert!(h.assets.stale, "the expansion follows the entries");
    h.mark_changed();
    h.refresh_tree_if_needed();
    assert!(!h.assets.stale, "one refresh covers the burst");
}

#[test]
fn groups_fold_and_unfold() {
    let _guard = crate::test_support::lock();
    let mut h = expandable_hook();
    h.refresh_tree_if_needed();
    let folded = h.tree_rows().len();
    assert_eq!(folded, h.assets.groups.len(), "headers only");

    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert!(h.tree_rows().len() > folded, "the group unfolded");
    h.apply_panel(PanelAction::ToggleGroup(0), &mut world_with_fields());
    assert_eq!(h.tree_rows().len(), folded, "and folded again");
}

// A tree row click mirrors a viewport pick: plain replaces the selection and
// opens the clicked entry's edit form; with shift held it toggles membership
// instead.
#[test]
fn tree_row_click_selects_and_opens_the_form() {
    let mut world = world_with_fields();
    let mut h = hook(vec![entry("box", "Sprite"), entry("cam", "Camera3D")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    click_row(&mut h, "box", &mut world);
    assert_eq!(active(&h).as_deref(), Some("box"));
    assert!(
        h.open[PanelKey::Assets],
        "the assets UI comes up around the form"
    );
    assert_eq!(
        h.form.target,
        entry_target(&h, 0),
        "the form targets the clicked entry"
    );

    let (g, i) = row_of(&h, "cam");
    h.apply_panel(PanelAction::ExtendRow(g, i), &mut world);
    assert_eq!(selected(&h), ["box", "cam"], "a modified click adds");
    h.apply_panel(PanelAction::ExtendRow(g, i), &mut world);
    assert_eq!(selected(&h), ["box"], "a second one removes");
}

//! The Content panel's grid (`hook/edit/content.rs`): the visual assets it
//! lists, the filter over them, and what a cell click selects. Carrying a cell
//! out into the world is `tests/drag/content_tests.rs`.

use concinnity_cook::authoring::registry::RegisteredType;

use crate::editor::hook::edit::content::VISUAL_TYPES;
use crate::editor::hook::tests::fixtures::{active, entry, hook};

use crate::editor::panels::content_panel::ContentAction;
use crate::editor::panels::registry::PanelKey;

// The Content grid over a world with visual assets: cells list them with
// icon fallbacks (no thumbnails baked in tests), the type button narrows, the
// search query ranks, and a cell click selects the asset.
#[test]
fn content_grid_lists_filters_and_selects_visual_assets() {
    let mut h = hook(vec![
        serde_json::json!({
            "type": "Texture",
            "args": { "$id": "brick_tex", "generator": "brick", "resolution": 32 }
        }),
        serde_json::json!({
            "type": "Material", "args": { "$id": "brick_mat", "roughness": 0.5 }
        }),
        entry("note", "TextLabel"),
    ]);
    h.open[PanelKey::Content] = true;
    h.assets.stale = true;
    h.refresh_tree_if_needed();

    h.refresh_content_cells();
    let names = |h: &crate::editor::hook::EditorHook| -> Vec<String> {
        h.content.items.iter().map(|(n, _)| n.clone()).collect()
    };
    assert_eq!(
        names(&h),
        ["brick_mat", "brick_tex"],
        "sorted, TextLabel absent"
    );
    assert!(
        h.content.cells.iter().all(|c| c.image.is_none()),
        "no baked thumbnails in tests: every cell falls back to its icon"
    );

    // The type button narrows to one kind.
    while h.content_type_caption() != "Material" {
        h.cycle_content_type();
    }
    h.refresh_content_cells();
    assert_eq!(
        h.content.items,
        [("brick_mat".to_string(), RegisteredType::Material)]
    );
    h.content.type_chip = 0;

    // The search field ranks name matches.
    h.content.filter = "tex".to_string();
    h.refresh_content_cells();
    assert_eq!(names(&h)[0], "brick_tex", "the query's best match leads");

    // A cell click selects the asset by name.
    h.apply_content_action(ContentAction::Select(0));
    assert_eq!(active(&h).as_deref(), Some("brick_tex"));
    h.refresh_content_cells();
    assert!(
        h.content.cells[0].selected,
        "the grid highlights the selection"
    );
}

// The type button cycles through "All" and then every visual type, each captioned
// with its registered name.
#[test]
fn every_visual_type_captions_to_its_own_name() {
    let mut h = hook(vec![]);
    assert_eq!(h.content_type_caption(), "All");
    for ty in VISUAL_TYPES {
        h.cycle_content_type();
        assert_eq!(h.content_type_caption(), ty.as_str());
    }
    h.cycle_content_type();
    assert_eq!(h.content_type_caption(), "All");
}

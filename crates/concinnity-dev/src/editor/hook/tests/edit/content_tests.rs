//! The Content panel's grid (`hook/edit/content.rs`): the visual assets it
//! lists, the filter over them, and what a cell click selects. Carrying a cell
//! out into the world is `tests/drag/content_tests.rs`.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use crate::editor::hook::edit::content::VISUAL_TYPES;
use crate::editor::hook::tests::fixtures::{entry, hook};

use crate::editor::hook::tests::fixtures::active;
use crate::editor::panels::content_panel;

// The Content grid over a world with visual assets: cells list them with
// icon fallbacks (no thumbnails baked in tests), the type chip narrows, the
// search query ranks, and a cell click selects the asset.
#[test]
fn content_grid_lists_filters_and_selects_visual_assets() {
    let mut world = World::new();
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
    h.content_open = true;
    h.tree_stale = true;
    h.refresh_tree_if_needed();

    let (cells, total) = h.content_cells(&world);
    assert_eq!(total, 2, "only the visual types are listed");
    let names: Vec<&str> = cells.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        ["brick_mat", "brick_tex"],
        "sorted, TextLabel absent"
    );
    assert!(
        cells.iter().all(|c| c.thumb.is_none()),
        "no baked thumbnails in tests: every cell falls back to its icon"
    );

    // The type chip narrows to one kind.
    while h.content_type_caption() != "Material" {
        h.cycle_content_type();
    }
    let (cells, total) = h.content_cells(&world);
    assert_eq!((cells.len(), total), (1, 1));
    assert_eq!(cells[0].asset_type, RegisteredType::Material);
    h.content_type = 0;

    // The search field ranks name matches.
    world.push_identified(
        content_panel::SEARCH_INPUT,
        TextInput {
            content: "tex".to_string(),
            ..Default::default()
        },
    );
    let (cells, _) = h.content_cells(&world);
    assert_eq!(cells[0].name, "brick_tex", "the query's best match leads");

    // A cell click selects the asset by name.
    h.apply_content_action(
        crate::editor::panels::content_panel::ContentAction::SelectCell(0),
        &mut world,
        [0.0, 0.0],
    );
    assert_eq!(active(&h).as_deref(), Some("brick_tex"));
    let (cells, _) = h.content_cells(&world);
    assert!(cells[0].selected, "the grid highlights the selection");
}

// The type chip cycles through "All" and then every visual type, each captioned
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

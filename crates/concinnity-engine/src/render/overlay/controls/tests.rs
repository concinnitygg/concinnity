// The widgets as the overlay draws them: only what is in view, clipped to the
// widget, and a popup on its screen's popup layer.

use alloc::vec::Vec;

use super::*;
use crate::components::{
    Binding, Button, Checkbox, EntryMark, Icon, IconPart, ListToggle, Menu, MenuEntry, Modal,
    UiItem, UiState, UiValue,
};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::TextDrawCall;
use crate::render::overlay::test_fonts::{FONT, loaded_fonts};
use crate::render::overlay::{OverlayFrame, OverlaySystem, maps};

const SCREEN: AssetId = AssetId(50);

fn assets() -> OverlayAssets {
    OverlayAssets {
        fonts: loaded_fonts(),
        sprite_texture_slots: maps::TextureSlots::new(),
        clip_rects: maps::ClipRects::new(),
        initial_viewport: (1280.0, 720.0),
    }
}

fn style() -> WidgetStyle {
    WidgetStyle {
        font: Some(FONT),
        ..Default::default()
    }
}

// One frame of the overlay over `world`.
fn build(world: &mut World) -> OverlayFrame {
    let a = assets();
    OverlaySystem::new().build_frame(&mut world.context(), &a, 0.0)
}

// A world whose `files` list holds `n` rows, shown by a 200x100 list of
// 20-pixel rows.
fn list_world(n: usize) -> World {
    let mut world = World::new();
    let mut state = UiState::default();
    let items = (0..n)
        .map(|i| UiItem::new(&alloc::format!("row {i}")))
        .collect();
    state.set_list("files", items);
    world.insert_resource(state);
    world.push(List {
        items: "files".into(),
        width: 200.0,
        height: 100.0,
        row_height: 20.0,
        style: style(),
        ..Default::default()
    });
    world
}

#[test]
fn a_list_draws_only_its_rows_in_view_clipped_to_itself() {
    let few = build(&mut list_world(7)).calls;
    let many = build(&mut list_world(5000)).calls;
    assert_eq!(few.len(), many.len(), "5000 rows draw what 7 do");
    assert!(
        many.iter()
            .all(|c| c.clip_rect == Some([0.0, 0.0, 200.0, 100.0]))
    );
}

// A world whose `files` list holds `n` items, shown by a 300x200 grid of
// 90-pixel cells.
fn grid_world(n: usize) -> World {
    let mut world = list_world(n);
    world.remove_all::<List>();
    world.push(Grid {
        items: "files".into(),
        width: 300.0,
        height: 200.0,
        cell_width: 90.0,
        cell_height: 90.0,
        gap: 10.0,
        style: style(),
        ..Default::default()
    });
    world
}

#[test]
fn a_grid_draws_only_its_cells_in_view_clipped_to_itself() {
    let few = build(&mut grid_world(9)).calls;
    let many = build(&mut grid_world(5000)).calls;
    assert_eq!(few.len(), many.len(), "5000 cells draw what 9 do");
    assert!(
        many.iter()
            .all(|c| c.clip_rect == Some([0.0, 0.0, 300.0, 200.0]))
    );
}

// The vertices a list of one row draws for `item`, by its toggles.
fn row_sprites(item: UiItem) -> usize {
    let mut world = list_world(0);
    if let Some(s) = world.resource_mut::<UiState>() {
        s.set_list("files", alloc::vec![item]);
    }
    let dot = IconPart::fill([4.0, 4.0, 8.0, 8.0], 0.5);
    for list in world.query_mut::<List>() {
        list.toggles = alloc::vec![ListToggle {
            on: Icon {
                parts: alloc::vec![dot, dot],
            },
            ..Default::default()
        }];
    }
    build(&mut world)
        .calls
        .iter()
        .map(|c| c.vertices.len())
        .sum()
}

#[test]
fn a_toggle_cell_draws_its_icon_only_while_on() {
    let plain = row_sprites(UiItem::new("a"));
    let off = row_sprites(UiItem {
        toggles: Some(0),
        ..UiItem::new("a")
    });
    let on = row_sprites(UiItem {
        toggles: Some(1),
        ..UiItem::new("a")
    });
    assert_eq!(off, plain, "no off icon, not hovered");
    assert!(on > plain, "the on icon draws");
}

// The vertices a button captioned `label` draws with `icon`.
fn button_vertices(label: &str, icon: Icon) -> usize {
    let mut world = World::new();
    world.push(Button {
        label: label.into(),
        icon,
        style: style(),
        ..Default::default()
    });
    build(&mut world)
        .calls
        .iter()
        .map(|c| c.vertices.len())
        .sum()
}

#[test]
fn an_icon_button_draws_its_icon_in_place_of_its_caption() {
    let dot = || Icon {
        parts: alloc::vec![IconPart::fill([4.0, 4.0, 8.0, 8.0], 0.5)],
    };
    assert!(
        button_vertices("", dot()) > button_vertices("", Icon::default()),
        "the icon draws"
    );
    assert!(
        button_vertices("Raise", Icon::default()) > button_vertices("", Icon::default()),
        "a plain button writes its caption"
    );
    assert_eq!(
        button_vertices("Raise", dot()),
        button_vertices("", dot()),
        "an icon button writes none"
    );
}

// What a scrolling layout holds draws only inside it, and its scrollbar
// shows while its children overflow it.
#[test]
fn a_scrolling_layout_clips_what_it_holds() {
    let mut world = World::new();
    let rows: Vec<_> = (0..4)
        .map(|i| {
            let id = AssetId(60 + i);
            world.push_identified(
                id,
                Sprite {
                    width: 50.0,
                    height: 40.0,
                    tint: [1.0; 4],
                    ..Default::default()
                },
            );
            crate::components::LayoutChild::fit(id)
        })
        .collect();
    world.push_identified(
        AssetId(70),
        Layout {
            children: rows,
            scroll: true,
            ..Default::default()
        },
    );
    world.push(Layout {
        children: alloc::vec![crate::components::LayoutChild {
            item: Some(Ref::new(AssetId(70))),
            width: crate::components::LayoutSize::Fixed(120.0),
            height: crate::components::LayoutSize::Fixed(100.0),
        }],
        ..Default::default()
    });
    let calls = build(&mut world).calls;
    let clipped = calls
        .iter()
        .filter(|c| c.clip_rect == Some([0.0, 0.0, 120.0, 100.0]))
        .count();
    assert_eq!(clipped, 4, "every row, clipped to the layout");
    assert_eq!(calls.len(), 6, "and the scrollbar's track and thumb");
    let layout = world.query::<Layout>().find(|l| l.scroll).unwrap();
    assert_eq!(layout.content_height, 160.0);
}

#[test]
fn an_open_menu_draws_on_its_screens_menu_layer() {
    let mut world = World::new();
    let mut state = UiState::default();
    state.set("open", UiValue::Bool(false));
    world.insert_resource(state);
    world.insert_resource(ScreenStack {
        layers: alloc::collections::BTreeMap::from([(SCREEN, 8)]),
        open: alloc::vec![SCREEN],
        pauses_world: false,
        captures_input: true,
    });
    world.push(Menu {
        open: Some(Binding::State("open".into())),
        entries: alloc::vec![MenuEntry {
            label: "Quit".into(),
            ..Default::default()
        }],
        style: style(),
        screen: Some(Ref::new(SCREEN)),
        ..Default::default()
    });
    assert!(build(&mut world).calls.is_empty(), "closed");

    world
        .resource_mut::<UiState>()
        .unwrap()
        .set("open", UiValue::Bool(true));
    let calls: Vec<TextDrawCall> = build(&mut world).calls;
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|c| c.layer == Popup::Menu.over(8)));
}

// The vertices an open menu of one entry draws, its mark bound to `on`.
fn menu_vertices(checked: Option<bool>) -> usize {
    let mut world = World::new();
    let mut state = UiState::default();
    state.set("open", UiValue::Bool(true));
    if let Some(on) = checked {
        state.set("on", UiValue::Bool(on));
    }
    world.insert_resource(state);
    world.push(Menu {
        open: Some(Binding::State("open".into())),
        entries: alloc::vec![MenuEntry {
            label: "Grid".into(),
            mark: checked.map(|_| EntryMark::Check(Binding::State("on".into()))),
            ..Default::default()
        }],
        style: style(),
        ..Default::default()
    });
    build(&mut world)
        .calls
        .iter()
        .map(|c| c.vertices.len())
        .sum()
}

#[test]
fn a_checked_entry_draws_its_mark_filled_only_while_on() {
    let plain = menu_vertices(None);
    let off = menu_vertices(Some(false));
    let on = menu_vertices(Some(true));
    assert!(off > plain, "the empty mark draws");
    assert!(on > off, "and fills while on");
}

// A shown modal root draws a dim over its screen, under itself.
#[test]
fn a_modal_root_dims_its_screen() {
    let mut world = World::new();
    world.push_identified(AssetId(10), Sprite::default());
    world.push(Layout {
        modal: true,
        children: alloc::vec![crate::components::LayoutChild::fit(AssetId(10))],
        ..Default::default()
    });
    // The first frame lays the root out; the second stacks it.
    build(&mut world);
    let ranks: Vec<u32> = build(&mut world).calls.iter().map(|c| c.rank).collect();
    assert_eq!(ranks.len(), 2, "the dim and the sprite");
    assert!(
        ranks[0] < ranks[1],
        "the dim under what the root holds: {ranks:?}"
    );
    for layout in world.query_mut::<Layout>() {
        layout.visible = false;
    }
    build(&mut world);
    assert!(
        build(&mut world).calls.is_empty(),
        "hidden, no dim and no sprite"
    );
}

// A canvas draws its shapes panned into place, the ones past its edge left
// out, its text with the shape, clipped to itself.
#[test]
fn a_canvas_draws_the_shapes_that_reach_it() {
    let mut world = World::new();
    let mut state = UiState::default();
    state.edit_shapes("chart", |shapes| {
        for x in [0.0, 500.0] {
            shapes.push(crate::components::UiShape {
                rect: [x, 0.0, 40.0, 20.0],
                fill: [1.0; 4],
                text: "Card".into(),
                ..Default::default()
            });
        }
    });
    world.insert_resource(state);
    world.push(Canvas {
        shapes: "chart".into(),
        width: 200.0,
        height: 100.0,
        style: style(),
        ..Default::default()
    });
    let calls = build(&mut world).calls;
    assert!(
        calls.iter().all(|c| c.clip_rect.is_some()),
        "clipped to itself"
    );
    let drawn: usize = calls.iter().map(|c| c.vertices.len()).sum();
    for c in world.query_mut::<Canvas>() {
        c.pan = [480.0, 0.0];
    }
    let panned: usize = build(&mut world)
        .calls
        .iter()
        .map(|c| c.vertices.len())
        .sum();
    assert_eq!(drawn, panned, "one card in view either way");
    assert!(drawn > 4, "the card and its text");
}

// A shape's text is cut to the room it has at its own text scale: the text a
// narrow shape cuts at full size fits once drawn at a quarter of it, and a
// shape too narrow for any text draws none.
#[test]
fn canvas_text_is_cut_at_its_scale() {
    let label = |scale: f32, width: f32| {
        let mut world = World::new();
        let mut state = UiState::default();
        state.edit_shapes("chart", |shapes| {
            shapes.push(crate::components::UiShape {
                rect: [0.0, 0.0, width, 20.0],
                text: "a long card title".into(),
                text_scale: scale,
                ..Default::default()
            });
        });
        world.insert_resource(state);
        world.push(Canvas {
            shapes: "chart".into(),
            width: 200.0,
            height: 100.0,
            style: style(),
            ..Default::default()
        });
        let mut scratch = WidgetScratch::default();
        let a = assets();
        let spaces = ScreenSpaces::default();
        let control = ControlFrame {
            assets: &a,
            stack: None,
            panels: &PanelStack::default(),
            spaces: &spaces,
            viewport: [1280.0, 720.0],
            default_atlas_slot: None,
            caret_visible: false,
        };
        let ctx = world.context();
        let canvas = ctx.query::<Canvas>().next().unwrap();
        canvas::draw(canvas, ctx.resource::<UiState>(), &control, &mut scratch);
        scratch.labels.first().map(|l| l.content.clone())
    };
    let full = label(1.0, 80.0).expect("the text draws");
    assert!(full.ends_with("..."), "cut at full size: {full}");
    assert_eq!(label(0.25, 80.0).as_deref(), Some("a long card title"));
    assert_eq!(label(1.0, 10.0), None, "no room, no text");
}

#[test]
fn a_widget_on_a_closed_screen_draws_nothing() {
    let mut world = list_world(3);
    world.insert_resource(ScreenStack::default());
    for list in world.query_mut::<List>() {
        list.screen = Some(Ref::new(SCREEN));
    }
    assert!(build(&mut world).calls.is_empty());
}

const LONG: &str = "a line of words far too wide for the narrow widget that has to show it";

// A world holding a list whose rows overflow it and an open dialog whose
// message wraps.
fn overflowing_world() -> World {
    let mut world = list_world(500);
    if let Some(state) = world.resource_mut::<UiState>() {
        state.set_list("long", alloc::vec![UiItem::new(LONG); 8]);
        state.set("ask", UiValue::Bool(true));
    }
    world.push(List {
        items: "long".into(),
        y: 120.0,
        width: 120.0,
        height: 100.0,
        row_height: 20.0,
        style: style(),
        ..Default::default()
    });
    world.push(Modal {
        open: Some(Binding::State("ask".into())),
        title: LONG.into(),
        message: LONG.into(),
        width: 200.0,
        style: style(),
        ..Default::default()
    });
    world
}

#[test]
fn overflowing_widget_text_is_cut_or_wrapped_before_it_is_shaped() {
    let mut world = overflowing_world();
    // A built frame lays the dialog out.
    build(&mut world);
    let mut scratch = WidgetScratch::default();
    let a = assets();
    let spaces = ScreenSpaces::default();
    let control = ControlFrame {
        assets: &a,
        stack: None,
        panels: &PanelStack::default(),
        spaces: &spaces,
        viewport: [1280.0, 720.0],
        default_atlas_slot: None,
        caret_visible: false,
    };
    let ctx = world.context();
    let modal = ctx.query::<Modal>().next().unwrap();
    popups::draw_modal(modal, &control, &mut scratch);
    let message = scratch
        .labels
        .iter()
        .find(|l| l.content.contains('\n'))
        .expect("the message broke onto lines");
    assert!(message.content.lines().count() > 1);
    let title = &scratch.labels[0];
    assert!(title.content.ends_with("..."), "the title was cut");
    assert!(
        scratch
            .labels
            .iter()
            .all(|l| l.wrap_width == 0.0 && l.max_lines == 0),
        "no label is left for the shaper to wrap"
    );
}

#[test]
fn a_steady_frame_of_widgets_allocates_nothing() {
    let mut world = overflowing_world();
    world.push(TextArea {
        x: 300.0,
        language: crate::components::TextLanguage::Hlsl,
        content: "float4 main() : SV_Target { return 1; } // done\nint x;".into(),
        style: style(),
        read_at: Some((None, None)),
        ..Default::default()
    });
    for area in world.query_mut::<TextArea>() {
        *area.editor = crate::presentation::text_edit::TextEditor::from_text(&area.content)
            .highlighted(&crate::presentation::text_edit::highlight::HLSL);
    }
    let mut toasts = Toasts {
        style: style(),
        ..Default::default()
    };
    toasts.push(&crate::components::Toast::new(
        LONG,
        crate::components::ToastTone::Info,
    ));
    toasts.shown[0].left = 1.0e9;
    world.push(toasts);
    let a = assets();
    let mut sys = OverlaySystem::new();
    for _ in 0..32 {
        let frame = sys.build_frame(&mut world.context(), &a, 0.0);
        world.insert_resource(crate::render::overlay::OverlayRecycle(frame.calls));
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut min = u64::MAX;
    while min > 0 && std::time::Instant::now() < deadline {
        let before = concinnity_core::memory::alloc_count().expect("the binary tracks its heap");
        let frame = sys.build_frame(&mut world.context(), &a, 0.0);
        world.insert_resource(crate::render::overlay::OverlayRecycle(frame.calls));
        let after = concinnity_core::memory::alloc_count().expect("the allocator stays installed");
        min = min.min(after - before);
    }
    assert_eq!(min, 0, "a steady widget frame allocated {min} times");
}

#[test]
fn a_panel_brought_forward_draws_over_the_one_added_after_it() {
    use crate::components::{Layout, LayoutChild, Sprite};
    let mut world = World::new();
    for (id, holds, rank) in [(1, 10, 5), (2, 20, 0)] {
        world.push_identified(
            AssetId(holds),
            Sprite {
                width: 10.0,
                height: 10.0,
                tint: [1.0; 4],
                ..Default::default()
            },
        );
        world.push_identified(
            AssetId(id),
            Layout {
                movable: true,
                children: alloc::vec![LayoutChild {
                    item: Some(Ref::new(AssetId(holds))),
                    ..Default::default()
                }],
                rank,
                ..Default::default()
            },
        );
    }
    // The first frame lays the panels out; the second stacks them.
    build(&mut world);
    let ranks: Vec<u32> = build(&mut world).calls.iter().map(|c| c.rank).collect();
    assert_eq!(ranks, [1, 2], "the ranked panel's sprite draws last");
}

#[test]
fn a_widget_draws_on_its_screens_layer() {
    let mut world = World::new();
    world.push_identified(
        AssetId(7),
        Checkbox {
            label: "on".into(),
            width: 100.0,
            height: 24.0,
            style: style(),
            screen: Some(Ref::new(SCREEN)),
            ..Default::default()
        },
    );
    world.insert_resource(ScreenStack {
        layers: alloc::collections::BTreeMap::from([(SCREEN, 3)]),
        open: alloc::vec![SCREEN],
        ..Default::default()
    });
    let frame = build(&mut world);
    assert!(!frame.calls.is_empty());
    assert!(frame.calls.iter().all(|c| c.layer == 3));
}

#[test]
fn a_hidden_layout_draws_nothing_it_holds() {
    use crate::components::{Layout, LayoutChild, Sprite};
    let mut world = World::new();
    world.push_identified(
        AssetId(10),
        Sprite {
            width: 10.0,
            height: 10.0,
            tint: [1.0; 4],
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(11),
        Checkbox {
            label: "on".into(),
            style: style(),
            ..Default::default()
        },
    );
    let children = [10, 11]
        .map(|id| LayoutChild {
            item: Some(Ref::new(AssetId(id))),
            ..Default::default()
        })
        .to_vec();
    world.push(Layout {
        children,
        visible: false,
        ..Default::default()
    });
    assert!(build(&mut world).calls.is_empty());
}

#[test]
fn a_marked_line_draws_a_dot_in_the_gutter() {
    let area = |marks| {
        let mut world = World::new();
        world.push(TextArea {
            content: "one\ntwo".into(),
            style: style(),
            read_at: Some((None, None)),
            marks,
            ..Default::default()
        });
        for a in world.query_mut::<TextArea>() {
            *a.editor = crate::presentation::text_edit::TextEditor::from_text(&a.content);
        }
        let mut sys = OverlaySystem::new();
        let a = assets();
        sys.build_frame(&mut world.context(), &a, 0.0);
        sys.build_frame(&mut world.context(), &a, 0.0).calls.len()
    };
    let mark = crate::components::LineMark {
        line: 1,
        color: [1.0, 0.0, 0.0],
    };
    assert_eq!(area(alloc::vec![mark]), area(Vec::new()) + 1);
}

//! The viewport's marks: one window-sized canvas, drawn under every panel and
//! taking no input, that the editor writes its viewport furniture into each
//! frame -- the start screen's fade and loading cover, the billboards, the
//! selection rings, the gizmo, and the marquee, in that order, so each draws
//! over the ones before it.

use concinnity_engine::components::{Canvas, Layout, LayoutAnchor, LayoutSize, UiShape};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::ui::{UiBuilder, state};

// The shape list the canvas draws.
pub(crate) const SHAPES: &str = "editor.viewport";

// Build the canvas, filling a root of its own, returning the canvas.
pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<AssetId> {
    let canvas = ui.canvas(Canvas {
        shapes: SHAPES.to_string(),
        passthrough: true,
        clip: false,
        ..Default::default()
    })?;
    let canvas_id = canvas.item?.id();
    ui.layout(Layout {
        children: vec![canvas],
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fill(1.0),
        anchor: LayoutAnchor::TopLeft,
        ..Default::default()
    })?;
    Some(canvas_id)
}

// Hand `shapes` to the canvas, taking last frame's list back in its place, so
// the two lists trade storage and a steady frame allocates nothing.
pub(crate) fn swap_in(world: &mut World, shapes: &mut Vec<UiShape>) {
    state::with_state(world, |s| {
        s.edit_shapes(SHAPES, |held| std::mem::swap(held, shapes));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    #[test]
    fn the_marks_canvas_takes_no_input_and_trades_lists() {
        let mut world = World::new();
        let id = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let canvas = world.get_by_id::<Canvas>(id).unwrap();
        assert!(canvas.passthrough && !canvas.clip);
        let mut shapes = vec![UiShape::default()];
        swap_in(&mut world, &mut shapes);
        assert!(shapes.is_empty(), "last frame's list came back");
        assert_eq!(world.resource::<UiState>().unwrap().shapes(SHAPES).len(), 1);
    }
}

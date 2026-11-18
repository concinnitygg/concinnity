//! `EditorHook`: the viewport marks drive. Each frame writes the viewport's
//! furniture into the marks canvas (`viewport/marks.rs`) bottom to top: the
//! start screen's fade and loading cover, then, while a world is being edited,
//! the billboards, the drag-out ghost, the selection rings, the gizmo, and the
//! marquee.

use concinnity_engine::components::UiShape;
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::ui::shapes::Shapes;
use crate::editor::viewport::marks;

// What the drive writes into between frames: the shape list the canvas
// trades with, and the loading caption.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct MarksScratch {
    shapes: Vec<UiShape>,
    caption: String,
}

impl EditorHook {
    // `shown` is whether the HUD is up over a real viewport.
    pub(in crate::editor::hook) fn drive_viewport_marks(
        &mut self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
    ) {
        let mut scratch = std::mem::take(&mut self.marks);
        let mut shapes = Shapes::new(&mut scratch.shapes);
        if shown {
            self.push_cinematic(&mut shapes, vp);
            self.push_loading(&mut shapes, &mut scratch.caption);
        }
        // The viewport overlays belong to a session editing a world: the
        // start screen shows the previewed world plainly.
        if shown && !self.start_mode && !self.sim.playing() {
            if self.show_billboards {
                self.push_billboards(world, vp, &mut shapes);
            }
            self.push_content_ghost(world, vp, &mut shapes);
            self.push_rings(world, vp, &mut shapes);
            self.push_gizmo(world, vp, &mut shapes);
            self.push_marquee(&mut shapes);
        }
        shapes.finish();
        marks::swap_in(world, &mut scratch.shapes);
        self.marks = scratch;
    }
}

// src/editor/hook/shape_drag.rs
//
// EditorHook: a CharacterShape slider drag. The press on a slider takes the
// value under the cursor; while the button is held the value follows the
// cursor and the edited shape is re-resolved against the live pose every
// frame (no rebuild); releasing commits the working values to the entry as
// ONE edit, so undo reverts the whole drag; Escape restores the start state.

use super::*;
use crate::assets::{CharacterCapsule, CharacterShape};
use crate::editor::character_shape::{ShapeValues, SliderRow};
use crate::editor::widget_slider;
use crate::gfx::shape_preview;

pub(super) struct ShapeDrag {
    pub(super) slider: usize,
    rect: [f32; 4],
    shape_idx: usize,
    mesh: String,
    sliders: Vec<SliderRow>,
    joints: Vec<String>,
    capsule: Option<CharacterCapsule>,
    start: ShapeValues,
    pub(super) values: ShapeValues,
}

impl EditorHook {
    pub(super) fn begin_shape_drag(
        &mut self,
        data: &super::character_shape_edit::ShapeData,
        slider: usize,
        rect: [f32; 4],
        mouse: [f32; 2],
        world: &mut World,
    ) {
        let Some(b) = &data.binding else {
            return;
        };
        let Some(shape_idx) = b.shape_idx else {
            return;
        };
        let start = self.shape_values(shape_idx);
        let mut drag = ShapeDrag {
            slider,
            rect,
            shape_idx,
            mesh: b.mesh.clone(),
            sliders: data.derived.sliders.clone(),
            joints: data.target.joint_names.clone(),
            capsule: self.mesh_capsule(&b.mesh),
            values: start.clone(),
            start,
        };
        drag.follow(mouse[0]);
        drag.preview(world);
        self.shape_drag = Some(drag);
    }

    // Per-frame drive: follow the cursor while the button is held, cancel on
    // Escape, commit on release.
    pub(super) fn drive_shape_drag(&mut self, input: &FrameInput, world: &mut World) {
        if input.escape {
            if let Some(drag) = self.shape_drag.take() {
                drag.restore(world);
            }
            return;
        }
        let Some(drag) = &mut self.shape_drag else {
            return;
        };
        if input.left_button_down {
            if drag.follow(input.mouse_x) {
                drag.preview(world);
            }
            return;
        }
        let Some(drag) = self.shape_drag.take() else {
            return;
        };
        if drag.values != drag.start {
            self.commit_shape(drag.shape_idx, &drag.values);
        }
    }
}

impl ShapeDrag {
    // Take the value under cursor x; `true` when it changed.
    fn follow(&mut self, mx: f32) -> bool {
        let row = &self.sliders[self.slider];
        let value = widget_slider::value_at(self.rect, mx, row.kind.range());
        let before = self.values.clone();
        self.values.set(row, value, &self.joints);
        self.values != before
    }

    fn preview(&self, world: &mut World) {
        self.apply_values(world, &self.values);
    }

    fn restore(&self, world: &mut World) {
        self.apply_values(world, &self.start);
    }

    fn apply_values(&self, world: &mut World, values: &ShapeValues) {
        let Some(id) = crate::ecs::asset_id::lookup(&self.mesh) else {
            return;
        };
        let Some(handle) = shape_preview::mesh_handle(world, id) else {
            return;
        };
        let shape = CharacterShape {
            target: Some(handle),
            sliders: values.sliders.clone(),
            proportions: values.proportions.clone(),
            ..Default::default()
        };
        shape_preview::apply(world, &shape, self.capsule.as_ref());
    }
}

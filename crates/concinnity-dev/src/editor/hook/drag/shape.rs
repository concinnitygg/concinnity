//! `EditorHook`: a `CharacterShape` slider drag. The first change a held slider
//! sends starts the drag from the entry's values; each change re-resolves the
//! edited shape against the live pose (no rebuild); letting go commits the
//! working values to the entry as ONE edit, so undo reverts the whole drag.
//! Escape restores the start state, and the rest of that drag is ignored.

use concinnity_engine::components::{CharacterCapsule, CharacterShape, FrameInput};
use concinnity_engine::ecs::World;
use concinnity_engine::render::live_preview::shape as shape_preview;
use concinnity_host::thread::asset_id;

use crate::editor::hook::EditorHook;
use crate::editor::panels::character_shape::{ShapeValues, SliderRow};

pub(in crate::editor::hook) struct ShapeDrag {
    slider: usize,
    row: SliderRow,
    shape_idx: usize,
    mesh: String,
    joints: Vec<String>,
    capsule: Option<CharacterCapsule>,
    start: ShapeValues,
    pub(in crate::editor::hook) values: ShapeValues,
    cancelled: bool,
}

impl EditorHook {
    // Slider `slider` moved to `value` while held: preview the shape with it.
    pub(in crate::editor::hook) fn drag_shape_slider(
        &mut self,
        slider: usize,
        value: f32,
        world: &mut World,
    ) {
        if self.shape.drag.as_ref().is_none_or(|d| d.slider != slider) {
            self.shape.drag = self.begin_shape_drag(slider, world);
        }
        let Some(drag) = &mut self.shape.drag else {
            return;
        };
        if drag.cancelled || !drag.set(value) {
            return;
        }
        drag.preview(world);
        let shown = drag.values.get(&drag.row);
        if let Some(v) = self
            .shape
            .cache
            .as_mut()
            .and_then(|c| c.values.get_mut(slider))
        {
            *v = shown;
        }
    }

    fn begin_shape_drag(&self, slider: usize, world: &World) -> Option<ShapeDrag> {
        let data = self.shape_data(world);
        let binding = data.binding?;
        let shape_idx = binding.shape_idx?;
        let row = data.derived.sliders.get(slider)?.clone();
        let start = self.shape_values(shape_idx);
        Some(ShapeDrag {
            slider,
            row,
            shape_idx,
            capsule: self.mesh_capsule(&binding.mesh),
            mesh: binding.mesh,
            joints: data.target.joint_names,
            values: start.clone(),
            start,
            cancelled: false,
        })
    }

    // Slider `slider` was let go at `value`: commit the drag, if it changed
    // anything and was not cancelled.
    pub(in crate::editor::hook) fn release_shape_slider(&mut self, slider: usize, value: f32) {
        let Some(mut drag) = self.shape.drag.take() else {
            return;
        };
        if drag.cancelled || drag.slider != slider {
            return;
        }
        drag.set(value);
        if drag.values != drag.start {
            self.commit_shape(drag.shape_idx, &drag.values);
        }
    }

    // Escape cancels a drag in flight: the start state shows again.
    pub(in crate::editor::hook) fn drive_shape_drag(
        &mut self,
        input: &FrameInput,
        world: &mut World,
    ) {
        if !input.escape {
            return;
        }
        let Some(drag) = &mut self.shape.drag else {
            return;
        };
        if drag.cancelled {
            return;
        }
        drag.cancelled = true;
        drag.restore(world);
        self.shape.cache = None;
    }
}

impl ShapeDrag {
    // Set the dragged row to `value`; `true` when that changed it.
    fn set(&mut self, value: f32) -> bool {
        let before = self.values.get(&self.row);
        self.values.set(&self.row, value, &self.joints);
        (self.values.get(&self.row) - before).abs() > f32::EPSILON
    }

    fn preview(&self, world: &mut World) {
        self.apply_values(world, &self.values);
    }

    fn restore(&self, world: &mut World) {
        self.apply_values(world, &self.start);
    }

    fn apply_values(&self, world: &mut World, values: &ShapeValues) {
        let Some(id) = asset_id::lookup(&self.mesh) else {
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

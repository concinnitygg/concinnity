// src/snapshot.rs
//
// The owned per-frame snapshot the extraction phase fills from world state
// and the submission phase consumes. Self-contained by construction: no
// borrows into component storage, resources, or the backend, so a frame's
// draw inputs can outlive the world borrow that produced them and later
// cross a thread boundary. Buffers keep their capacity across frames; a
// steady-state extraction allocates nothing.

use crate::render_types::{LineVertex, TextDrawCall};
use crate::scene_flow::SceneControl;
use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};

type Mat4 = [[f32; 4]; 4];

/// Camera and frame-wide flags for one frame's draw.
#[derive(Clone, Copy, Debug)]
pub struct FrameScalars {
    pub elapsed: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    /// View matrix the frame draws with (rebased when a chunk world streams).
    pub view: Mat4,
    /// Camera position in the space the frame renders in.
    pub cam_pos: [f32; 3],
    pub view_mode: ViewMode,
    pub show: ShowFlags,
    pub world_hidden: bool,
    pub menu_active: bool,
}

impl Default for FrameScalars {
    fn default() -> Self {
        Self {
            elapsed: 0.0,
            fov_y_radians: core::f32::consts::FRAC_PI_4,
            near: 0.05,
            far: 200.0,
            view: concinnity_core::gfx::transform::IDENTITY,
            cam_pos: [0.0; 3],
            view_mode: ViewMode::default(),
            show: ShowFlags::default(),
            world_hidden: false,
            menu_active: false,
        }
    }
}

/// Window-interaction intents resolved during extraction, applied verbatim by
/// submission. `None` means "do not touch the current state this frame".
#[derive(Clone, Copy, Debug, Default)]
pub struct UiIntents {
    /// Hide the OS cursor while an in-engine cursor sprite is shown.
    pub cursor_hidden: bool,
    pub menu_mode: Option<bool>,
    pub camera_capture: Option<bool>,
}

/// Variable-length per-slot updates flattened into one values buffer plus
/// `(slot, range)` spans, so extraction copies into persistent storage and
/// submission replays one backend call per span.
#[derive(Debug, Default)]
pub struct SpanBuffer<T> {
    values: Vec<T>,
    spans: Vec<(usize, u32, u32)>,
}

impl<T: Copy> SpanBuffer<T> {
    pub fn clear(&mut self) {
        self.values.clear();
        self.spans.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Append one slot's values as a new span.
    pub fn push(&mut self, slot: usize, values: &[T]) {
        let start = self.values.len() as u32;
        self.values.extend_from_slice(values);
        self.spans.push((slot, start, values.len() as u32));
    }

    /// The spans in push order as `(slot, values)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[T])> {
        self.spans
            .iter()
            .map(|&(slot, start, len)| (slot, &self.values[start as usize..(start + len) as usize]))
    }
}

/// One recorded scene-visibility effect, replayed onto the backend at
/// submission in record order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneOp {
    SetFade(f32),
    Visibility { draw_idx: usize, visible: bool },
}

/// A [`SceneControl`] that records calls as [`SceneOp`]s instead of driving a
/// backend, so scene-flow logic can run during extraction.
pub struct SceneOpRecorder<'a>(pub &'a mut Vec<SceneOp>);

impl SceneControl for SceneOpRecorder<'_> {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.0.push(SceneOp::Visibility { draw_idx, visible });
    }

    fn set_fade(&mut self, fade: f32) {
        self.0.push(SceneOp::SetFade(fade));
    }
}

/// Everything one frame's draw consumes, extracted from world state.
#[derive(Default)]
pub struct RenderSnapshot {
    pub frame: FrameScalars,
    pub ui: UiIntents,
    /// Changed static draw-slot model matrices, in push order (a slot pushed
    /// twice keeps both entries; the last write wins on the backend).
    pub models: Vec<(u32, Mat4)>,
    /// Changed skinned-instance model matrices, in push order.
    pub skinned_models: Vec<(u32, Mat4)>,
    /// Updated skinned joint matrices, keyed by skinned instance index.
    pub poses: SpanBuffer<Mat4>,
    /// Updated morph-target weights, keyed by skinned instance index.
    pub morphs: SpanBuffer<f32>,
    /// The frame's overlay draw list (UI text + sprites), adopted whole from
    /// the overlay build.
    pub text_calls: Vec<TextDrawCall>,
    /// Expanded world-space line ribbons for this frame's camera.
    pub lines: Vec<LineVertex>,
    /// Scene fade / visibility effects recorded this frame.
    pub scene_ops: Vec<SceneOp>,
}

impl RenderSnapshot {
    /// Reset for a new frame's extraction, keeping buffer capacity.
    pub fn clear(&mut self) {
        self.frame = FrameScalars::default();
        self.ui = UiIntents::default();
        self.models.clear();
        self.skinned_models.clear();
        self.poses.clear();
        self.morphs.clear();
        self.text_calls.clear();
        self.lines.clear();
        self.scene_ops.clear();
    }
}

// The snapshot must stay owned data so a later render thread can take it.
const _: () = {
    const fn require_send<T: Send + 'static>() {}
    require_send::<RenderSnapshot>()
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_buffer_round_trips_slots_in_push_order() {
        let mut spans: SpanBuffer<u32> = SpanBuffer::default();
        assert!(spans.is_empty());
        spans.push(7, &[1, 2, 3]);
        spans.push(2, &[9]);
        let collected: Vec<(usize, Vec<u32>)> =
            spans.iter().map(|(slot, v)| (slot, v.to_vec())).collect();
        assert_eq!(collected, vec![(7, vec![1, 2, 3]), (2, vec![9])]);
    }

    #[test]
    fn span_buffer_clear_keeps_capacity() {
        let mut spans: SpanBuffer<u32> = SpanBuffer::default();
        spans.push(0, &[1, 2, 3, 4]);
        spans.push(1, &[5, 6]);
        let values_ptr = spans.values.as_ptr();
        spans.clear();
        assert!(spans.is_empty());
        spans.push(0, &[1, 2, 3]);
        assert_eq!(
            spans.values.as_ptr(),
            values_ptr,
            "values buffer reallocated"
        );
    }

    #[test]
    fn recorder_captures_fade_and_visibility_in_order() {
        let mut ops = Vec::new();
        {
            let mut recorder = SceneOpRecorder(&mut ops);
            recorder.set_fade(0.5);
            recorder.update_visibility(3, false);
            recorder.update_visibility(4, true);
        }
        assert_eq!(
            ops,
            vec![
                SceneOp::SetFade(0.5),
                SceneOp::Visibility {
                    draw_idx: 3,
                    visible: false
                },
                SceneOp::Visibility {
                    draw_idx: 4,
                    visible: true
                },
            ]
        );
    }

    #[test]
    fn snapshot_clear_resets_contents_and_keeps_capacity() {
        let mut snap = RenderSnapshot::default();
        snap.models.push((1, [[0.0; 4]; 4]));
        snap.scene_ops.push(SceneOp::SetFade(1.0));
        snap.frame.elapsed = 5.0;
        let models_ptr = snap.models.as_ptr();
        snap.clear();
        assert!(snap.models.is_empty());
        assert!(snap.scene_ops.is_empty());
        assert_eq!(snap.frame.elapsed, 0.0);
        snap.models.push((2, [[0.0; 4]; 4]));
        assert_eq!(
            snap.models.as_ptr(),
            models_ptr,
            "models buffer reallocated"
        );
    }
}

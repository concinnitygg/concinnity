//! `EditorHook`: the Planet panel's tool session state: the palette's tools
//! and brush, where the brush is on the ground and the rings drawn around it,
//! the stroke a drag is laying, and the brush slider being dragged.

use concinnity_engine::planet::{GeoPosition, GroundPick};

use crate::editor::planet_tools::ring::RING_POINTS;
use crate::editor::planet_tools::stroke::Spacer;
use crate::editor::planet_tools::{Setting, Tool, Tools};

/// A drag laying points for `tool`.
#[derive(Debug, Clone, Copy)]
pub(in crate::editor::hook) struct Stroke {
    pub(in crate::editor::hook) tool: Tool,
    pub(in crate::editor::hook) spacer: Spacer,
}

#[derive(Debug)]
pub(in crate::editor::hook) struct PlanetToolState {
    pub(in crate::editor::hook) tools: Tools,
    /// The ground under the cursor, while a tool is picked.
    pub(in crate::editor::hook) hover: Option<GroundPick>,
    /// The brush's edge, and where its falloff begins, around `hover`.
    pub(in crate::editor::hook) ring: [[f32; 3]; RING_POINTS],
    pub(in crate::editor::hook) inner: Option<[[f32; 3]; RING_POINTS]>,
    pub(in crate::editor::hook) stroke: Option<Stroke>,
    /// The points the stroke laid this frame, its storage kept between frames.
    pub(in crate::editor::hook) points: Vec<GeoPosition>,
    pub(in crate::editor::hook) sliding: Option<Setting>,
}

impl Default for PlanetToolState {
    fn default() -> Self {
        Self {
            tools: Tools::default(),
            hover: None,
            ring: [[0.0; 3]; RING_POINTS],
            inner: None,
            stroke: None,
            points: Vec::new(),
            sliding: None,
        }
    }
}

impl PlanetToolState {
    /// Whether a drag of the tool's is under way.
    pub(in crate::editor::hook) fn busy(&self) -> bool {
        self.stroke.is_some() || self.sliding.is_some()
    }

    /// Put the tool down, ending any stroke. The brush is kept.
    pub(in crate::editor::hook) fn put_down(&mut self) {
        self.tools.active = None;
        self.hover = None;
        self.stroke = None;
        self.sliding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn putting_the_tool_down_keeps_the_brush() {
        let mut s = PlanetToolState::default();
        s.tools.toggle(Tool::Lower);
        s.tools.brush.radius = 40.0;
        s.sliding = Some(Setting::Radius);
        assert!(s.busy());
        s.put_down();
        assert!(!s.busy());
        assert_eq!(s.tools.active, None);
        assert_eq!(s.tools.brush.radius, 40.0);
    }
}

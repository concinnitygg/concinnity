//! The Planet panel's tools: which one is picked, the brush it works with,
//! and the keys that pick and size them. Editor session state only, never
//! world data. `stroke` spaces a drag into points along the ground and `ring`
//! lays the brush's outline on it.

pub(crate) mod ring;
pub(crate) mod stroke;

use concinnity_engine::components::{InputKey, KeyPress};
use concinnity_engine::planet::GeoPosition;

use super::icons::{self, Glyph};
use super::panels::planet::List;

/// A tool of the Planet panel's palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tool {
    Raise,
    Lower,
    Smooth,
    Flatten,
    PaintGrass,
    PaintCover,
    FillWater,
    Erase,
}

impl Tool {
    /// Every tool, in palette order.
    pub(crate) const ALL: [Self; 8] = [
        Self::Raise,
        Self::Lower,
        Self::Smooth,
        Self::Flatten,
        Self::PaintGrass,
        Self::PaintCover,
        Self::FillWater,
        Self::Erase,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Raise => "Raise",
            Self::Lower => "Lower",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::PaintGrass => "Paint grass",
            Self::PaintCover => "Paint cover",
            Self::FillWater => "Fill water",
            Self::Erase => "Erase",
        }
    }

    /// The key that picks the tool. None of them is one of the editor's
    /// own shortcuts or the fly camera's movement keys.
    pub(crate) fn key(self) -> InputKey {
        match self {
            Self::Raise => InputKey::U,
            Self::Lower => InputKey::J,
            Self::Smooth => InputKey::M,
            Self::Flatten => InputKey::L,
            Self::PaintGrass => InputKey::G,
            Self::PaintCover => InputKey::C,
            Self::FillWater => InputKey::O,
            Self::Erase => InputKey::X,
        }
    }

    pub(crate) fn for_key(key: InputKey) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.key() == key)
    }

    pub(crate) fn glyph(self) -> Glyph {
        match self {
            Self::Raise => icons::RAISE,
            Self::Lower => icons::LOWER,
            Self::Smooth => icons::SMOOTH,
            Self::Flatten => icons::FLATTEN,
            Self::PaintGrass => icons::GRASS,
            Self::PaintCover => icons::COVER,
            Self::FillWater => icons::WATER,
            Self::Erase => icons::ERASE,
        }
    }

    /// The planet list the tool paints an entry of.
    pub(crate) fn list(self) -> Option<List> {
        match self {
            Self::PaintGrass => Some(List::Layers),
            Self::PaintCover => Some(List::Covers),
            _ => None,
        }
    }
}

/// A brush setting with a slider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Setting {
    Radius,
    Strength,
    Falloff,
}

impl Setting {
    pub(crate) const ALL: [Self; 3] = [Self::Radius, Self::Strength, Self::Falloff];

    pub(crate) fn caption(self) -> &'static str {
        match self {
            Self::Radius => "radius",
            Self::Strength => "strength",
            Self::Falloff => "falloff",
        }
    }
}

/// The smallest and largest brush radius, in meters.
pub(crate) const MIN_RADIUS: f32 = 0.5;
pub(crate) const MAX_RADIUS: f32 = 500.0;

// The radius one bracket key press scales by.
const RADIUS_STEP: f32 = 1.25;

/// What a tool works with under the cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Brush {
    /// Meters along the ground from the center to the edge.
    pub radius: f32,
    /// How hard one pass works, from 0 to 1.
    pub strength: f32,
    /// The share of the radius, from the edge inward, over which the brush
    /// fades out; 0 is a hard edge.
    pub falloff: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            radius: 8.0,
            strength: 0.5,
            falloff: 0.5,
        }
    }
}

impl Brush {
    /// Where `setting` sits along its slider, from 0 to 1. The radius
    /// slides on a log scale, so small brushes get as much travel as large.
    pub(crate) fn position(&self, setting: Setting) -> f32 {
        match setting {
            Setting::Radius => (self.radius / MIN_RADIUS).ln() / (MAX_RADIUS / MIN_RADIUS).ln(),
            Setting::Strength => self.strength,
            Setting::Falloff => self.falloff,
        }
    }

    /// Move `setting`'s slider to `t`, clamped to `[0, 1]`.
    pub(crate) fn set_position(&mut self, setting: Setting, t: f32) {
        let t = t.clamp(0.0, 1.0);
        match setting {
            Setting::Radius => {
                self.radius =
                    (MIN_RADIUS * (MAX_RADIUS / MIN_RADIUS).powf(t)).clamp(MIN_RADIUS, MAX_RADIUS);
            }
            Setting::Strength => self.strength = t,
            Setting::Falloff => self.falloff = t,
        }
    }

    pub(crate) fn value(&self, setting: Setting) -> f32 {
        match setting {
            Setting::Radius => self.radius,
            Setting::Strength => self.strength,
            Setting::Falloff => self.falloff,
        }
    }

    /// Grow the radius by `steps` bracket presses, shrinking for negative.
    pub(crate) fn grow(&mut self, steps: i32) {
        self.radius = (self.radius * RADIUS_STEP.powi(steps)).clamp(MIN_RADIUS, MAX_RADIUS);
    }
}

/// The palette's state: the picked tool, its brush, and the layer and cover
/// the paint tools lay.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Tools {
    pub active: Option<Tool>,
    pub brush: Brush,
    /// Index into the planet's `layers`.
    pub layer: usize,
    /// Index into the planet's `covers`.
    pub cover: usize,
}

impl Tools {
    /// Pick `tool`, or put it down when it is already picked.
    pub(crate) fn toggle(&mut self, tool: Tool) {
        self.active = (self.active != Some(tool)).then_some(tool);
    }

    /// Act on a key press: a tool's key toggles it and the brackets size the
    /// brush while a tool is picked. Whether the press was taken. Presses
    /// under a modifier are left to the editor's shortcuts.
    pub(crate) fn key(&mut self, press: KeyPress) -> bool {
        let m = press.mods;
        if m.ctrl || m.cmd || m.alt {
            return false;
        }
        match press.key {
            InputKey::LeftBracket if self.active.is_some() => self.brush.grow(-1),
            InputKey::RightBracket if self.active.is_some() => self.brush.grow(1),
            key => match Tool::for_key(key) {
                Some(_) if press.repeat => {}
                Some(tool) => self.toggle(tool),
                None => return false,
            },
        }
        true
    }

    /// The entry of `list` the paint tools lay.
    pub(crate) fn pick(&self, list: List) -> usize {
        match list {
            List::Layers => self.layer,
            List::Covers => self.cover,
        }
    }

    /// Move the pick of `list` to its next entry of `count`, wrapping.
    pub(crate) fn cycle(&mut self, list: List, count: usize) {
        let next = if count == 0 {
            0
        } else {
            (self.pick(list).min(count - 1) + 1) % count
        };
        match list {
            List::Layers => self.layer = next,
            List::Covers => self.cover = next,
        }
    }
}

/// The handler a tool's strokes are delivered to: the points the stroke laid
/// this frame, in order along it, with the brush they were laid with.
pub(crate) fn work(_tool: Tool, _brush: &Brush, _points: &[GeoPosition]) {}

#[cfg(test)]
mod tests;

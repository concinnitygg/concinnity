// src/editor/snap.rs
//
// Grid and angle snapping for viewport editing: the step math plus the
// editor's snap settings. The gizmo drag snaps its translate delta and its
// applied rotate angle, so a group drag keeps the members' relative offsets;
// holding Ctrl during a drag temporarily inverts the enabled state. Session
// state like panel layout: not undoable, not saved.

// The step presets the Preview panel rows cycle through. The console's /snap
// accepts any positive step.
pub(crate) const TRANSLATE_STEPS: [f32; 3] = [0.1, 0.5, 1.0];
pub(crate) const ROTATE_STEPS: [f32; 3] = [5.0, 15.0, 45.0];

// One snap family: whether it applies and the grid interval it rounds to
// (meters for translate, degrees for rotate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Snap {
    pub enabled: bool,
    pub step: f32,
}

impl Snap {
    // The step to apply this frame, or `None` when snapping is off. `invert`
    // (held Ctrl during a drag) flips the enabled state for that drag.
    pub(crate) fn active_step(&self, invert: bool) -> Option<f32> {
        (self.enabled != invert && self.step > 0.0 && self.step.is_finite()).then_some(self.step)
    }

    // Advance to the next preset (the smallest preset above the current step,
    // wrapping to the first). A console-set off-preset step lands on the next
    // sensible preset.
    pub(crate) fn cycle(&mut self, presets: &[f32]) {
        self.step = presets
            .iter()
            .copied()
            .find(|&p| p > self.step + 1e-6)
            .unwrap_or(presets[0]);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SnapSettings {
    pub translate: Snap,
    pub rotate: Snap,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            translate: Snap {
                enabled: false,
                step: 0.5,
            },
            rotate: Snap {
                enabled: false,
                step: 15.0,
            },
        }
    }
}

impl SnapSettings {
    // The console status line.
    pub(crate) fn describe(&self) -> String {
        let state = |s: &Snap| if s.enabled { "on" } else { "off" };
        format!(
            "snap: move {} m ({}), rotate {} deg ({})",
            self.translate.step,
            state(&self.translate),
            self.rotate.step,
            state(&self.rotate),
        )
    }
}

// `v` rounded to the nearest multiple of `step`. A degenerate step returns `v`
// unchanged.
pub(crate) fn snap_step(v: f32, step: f32) -> f32 {
    if step <= 0.0 || !step.is_finite() {
        return v;
    }
    (v / step).round() * step
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_step_rounds_to_the_nearest_multiple() {
        assert_eq!(snap_step(0.74, 0.5), 0.5);
        assert_eq!(snap_step(0.76, 0.5), 1.0);
        assert_eq!(snap_step(-0.74, 0.5), -0.5);
        assert_eq!(snap_step(-0.76, 0.5), -1.0);
        assert_eq!(snap_step(0.0, 0.5), 0.0);
        assert_eq!(snap_step(37.0, 15.0), 30.0);
    }

    #[test]
    fn snap_step_ignores_degenerate_steps() {
        assert_eq!(snap_step(0.3, 0.0), 0.3);
        assert_eq!(snap_step(0.3, -1.0), 0.3);
        assert_eq!(snap_step(0.3, f32::NAN), 0.3);
        assert_eq!(snap_step(0.3, f32::INFINITY), 0.3);
    }

    #[test]
    fn active_step_answers_only_when_effectively_enabled() {
        let on = Snap {
            enabled: true,
            step: 0.5,
        };
        let off = Snap {
            enabled: false,
            step: 0.5,
        };
        assert_eq!(on.active_step(false), Some(0.5));
        assert_eq!(on.active_step(true), None, "held Ctrl suspends snapping");
        assert_eq!(off.active_step(false), None);
        assert_eq!(off.active_step(true), Some(0.5), "held Ctrl enables it");
        let broken = Snap {
            enabled: true,
            step: 0.0,
        };
        assert_eq!(broken.active_step(false), None);
    }

    #[test]
    fn cycle_walks_the_presets_and_wraps() {
        let mut s = Snap {
            enabled: true,
            step: 0.1,
        };
        s.cycle(&TRANSLATE_STEPS);
        assert_eq!(s.step, 0.5);
        s.cycle(&TRANSLATE_STEPS);
        assert_eq!(s.step, 1.0);
        s.cycle(&TRANSLATE_STEPS);
        assert_eq!(s.step, 0.1, "past the last preset it wraps");
    }

    #[test]
    fn cycle_lands_an_off_preset_step_on_the_next_preset() {
        let mut s = Snap {
            enabled: true,
            step: 0.3,
        };
        s.cycle(&TRANSLATE_STEPS);
        assert_eq!(s.step, 0.5);
        let mut r = Snap {
            enabled: true,
            step: 90.0,
        };
        r.cycle(&ROTATE_STEPS);
        assert_eq!(r.step, 5.0, "above every preset it wraps to the first");
    }

    #[test]
    fn describe_reports_both_families() {
        let s = SnapSettings::default();
        assert_eq!(s.describe(), "snap: move 0.5 m (off), rotate 15 deg (off)");
    }
}

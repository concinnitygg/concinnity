// src/ui/focus.rs
//
// Focus model for cursor-free menu navigation: derives focusable targets from
// the active screen's hit regions and picks the next focus for a directional
// pulse. Pure geometry + action-string grouping; UiInputSystem owns the focus
// state and applies the resulting styling / actions to the world.

use crate::assets::NavDirection;
use crate::gfx::setting_action;

// Weight of the perpendicular offset in the directional score, so a target
// straight ahead beats a nearer one far off to the side (tabs above a row
// list, dialog choice columns).
const PERP_WEIGHT: f32 = 2.0;

// A focusable region, described by its index into UiInputSystem's region list
// and its current (reflowed) rectangle.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) index: usize,
    pub(crate) rect: [f32; 4],
    pub(crate) action: String,
}

// One focusable control derived from the candidates. `index` is the region
// fired on confirm; `setting` is the key a Left/Right pulse adjusts (a
// stepper, dropdown, or slider row) instead of moving focus. Confirm on a
// slider is inert without needing a flag here: its track region is driven by
// the drag pass, which the dispatch loop already skips.
#[derive(Debug, Clone)]
pub(crate) struct Target {
    pub(crate) index: usize,
    pub(crate) rect: [f32; 4],
    pub(crate) setting: Option<String>,
}

// The current focus: the focused region's index plus its last known rect. The
// rect re-anchors navigation when the region vanishes between pulses (its row
// collapsed, its setting was disabled, or it scrolled away), so focus resumes
// from where it was instead of snapping to the top.
#[derive(Debug, Clone)]
pub(crate) struct FocusRef {
    pub(crate) index: usize,
    pub(crate) rect: [f32; 4],
}

// Group the candidates into focus targets:
//   - a stepper's `setting:<key>:next` region is the row's target (confirm
//     cycles forward); its `:prev` twin is dropped -- Left sends the Prev op
//     directly, so the twin region is never needed;
//   - a dropdown `:open` region and a slider `:drag` region are value rows
//     (Left/Right adjust);
//   - everything else with an action is a plain activate target.
pub(crate) fn targets(candidates: &[Candidate]) -> Vec<Target> {
    candidates
        .iter()
        .filter_map(|c| {
            if setting_action::key_with_verb(&c.action, "prev").is_some() {
                return None;
            }
            if let Some(key) = setting_action::key_with_verb(&c.action, "next") {
                return Some(Target {
                    index: c.index,
                    rect: c.rect,
                    setting: Some(key.to_string()),
                });
            }
            if let Some(key) = setting_action::key_with_verb(&c.action, "open") {
                return Some(Target {
                    index: c.index,
                    rect: c.rect,
                    setting: Some(key.to_string()),
                });
            }
            if let Some(key) = setting_action::key_with_verb(&c.action, "drag") {
                return Some(Target {
                    index: c.index,
                    rect: c.rect,
                    setting: Some(key.to_string()),
                });
            }
            (!c.action.is_empty()).then_some(Target {
                index: c.index,
                rect: c.rect,
                setting: None,
            })
        })
        .collect()
}

fn center(rect: [f32; 4]) -> (f32, f32) {
    (rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5)
}

// The signed distance from `from` to `to` along `dir` (positive = ahead) and
// the perpendicular offset magnitude.
fn distances(from: (f32, f32), to: (f32, f32), dir: NavDirection) -> (f32, f32) {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    match dir {
        NavDirection::Up => (-dy, dx.abs()),
        NavDirection::Down => (dy, dx.abs()),
        NavDirection::Left => (-dx, dy.abs()),
        NavDirection::Right => (dx, dy.abs()),
    }
}

// The target nearest `from` in `dir` by the weighted directional score, or
// `None` when nothing lies in that direction. `skip` excludes the currently
// focused region.
fn nearest_in_direction(
    targets: &[Target],
    from: (f32, f32),
    dir: NavDirection,
    skip: Option<usize>,
) -> Option<usize> {
    targets
        .iter()
        .filter(|t| Some(t.index) != skip)
        .filter_map(|t| {
            let (ahead, perp) = distances(from, center(t.rect), dir);
            (ahead > 0.5).then_some((t.index, ahead + perp * PERP_WEIGHT))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

// The topmost-leftmost target: where focus lands on the first pulse.
pub(crate) fn initial(targets: &[Target]) -> Option<usize> {
    targets
        .iter()
        .min_by(|a, b| {
            let (ax, ay) = center(a.rect);
            let (bx, by) = center(b.rect);
            (ay, ax)
                .partial_cmp(&(by, bx))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| t.index)
}

// The region to focus after a directional pulse, or `None` when there is
// nothing to focus. With no current focus, any pulse lands on the
// topmost-leftmost target. Up/Down wrap past the ends; Left/Right clamp (a
// value row consumes Left/Right before this is called, and a tab bar's ends
// should not teleport across the screen).
pub(crate) fn navigate(
    targets: &[Target],
    current: Option<&FocusRef>,
    dir: NavDirection,
) -> Option<usize> {
    if targets.is_empty() {
        return None;
    }
    let Some(current) = current else {
        return initial(targets);
    };
    let from = center(current.rect);
    // Only skip the current region when it still exists: a vanished region's
    // rect is just an anchor point, and the target now nearest it may sit
    // exactly there.
    let skip = targets
        .iter()
        .any(|t| t.index == current.index)
        .then_some(current.index);
    if let Some(next) = nearest_in_direction(targets, from, dir, skip) {
        return Some(next);
    }
    match dir {
        // Wrap: past the bottom, focus the topmost target (and vice versa),
        // keeping the horizontal position via the perpendicular weight.
        NavDirection::Up | NavDirection::Down => {
            let flipped = match dir {
                NavDirection::Up => NavDirection::Down,
                _ => NavDirection::Up,
            };
            // The extreme target opposite the pulse: the one furthest in the
            // flipped direction.
            targets
                .iter()
                .filter(|t| Some(t.index) != skip)
                .filter_map(|t| {
                    let (ahead, perp) = distances(from, center(t.rect), flipped);
                    (ahead > 0.5).then_some((t.index, ahead - perp * PERP_WEIGHT))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(index, _)| index)
                .or(Some(current.index))
        }
        // Clamp: stay put.
        NavDirection::Left | NavDirection::Right => Some(current.index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(index: usize, x: f32, y: f32, action: &str) -> Candidate {
        Candidate {
            index,
            rect: [x, y, 200.0, 30.0],
            action: action.to_string(),
        }
    }

    fn focus(targets: &[Target], index: usize) -> FocusRef {
        let t = targets.iter().find(|t| t.index == index).unwrap();
        FocusRef {
            index,
            rect: t.rect,
        }
    }

    #[test]
    fn steppers_group_to_one_target_and_prev_twins_drop() {
        let t = targets(&[
            cand(0, 0.0, 0.0, "setting:vsync:prev"),
            cand(1, 60.0, 0.0, "setting:vsync:next"),
            cand(2, 0.0, 40.0, "setting:exposure:drag"),
            cand(3, 0.0, 80.0, "setting:window_mode:open"),
            cand(4, 0.0, 120.0, "setting:key_forward:rebind"),
            cand(5, 0.0, 160.0, "group:toggle:0"),
            cand(6, 0.0, 200.0, "screen:hide"),
            cand(7, 0.0, 240.0, ""),
        ]);
        let by_index: Vec<usize> = t.iter().map(|t| t.index).collect();
        assert_eq!(by_index, vec![1, 2, 3, 4, 5, 6], "prev + empty drop");
        assert_eq!(t[0].setting.as_deref(), Some("vsync"));
        assert_eq!(t[1].setting.as_deref(), Some("exposure"));
        assert_eq!(t[2].setting.as_deref(), Some("window_mode"));
        assert!(t[3].setting.is_none(), "a rebind row is not a value row");
    }

    #[test]
    fn first_pulse_lands_topmost_leftmost() {
        let t = targets(&[
            cand(0, 100.0, 200.0, "screen:hide"),
            cand(1, 0.0, 50.0, "quit"),
            cand(2, 300.0, 50.0, "scene:1"),
        ]);
        assert_eq!(navigate(&t, None, NavDirection::Down), Some(1));
        assert_eq!(initial(&t), Some(1));
    }

    #[test]
    fn vertical_list_walks_and_wraps() {
        let t = targets(&[
            cand(0, 0.0, 0.0, "a:1"),
            cand(1, 0.0, 50.0, "a:2"),
            cand(2, 0.0, 100.0, "a:3"),
        ]);
        let f0 = focus(&t, 0);
        assert_eq!(navigate(&t, Some(&f0), NavDirection::Down), Some(1));
        let f2 = focus(&t, 2);
        assert_eq!(
            navigate(&t, Some(&f2), NavDirection::Down),
            Some(0),
            "wraps to top"
        );
        assert_eq!(
            navigate(&t, Some(&f0), NavDirection::Up),
            Some(2),
            "wraps to bottom"
        );
    }

    #[test]
    fn horizontal_neighbors_reachable_and_clamped_at_the_ends() {
        // A tab bar: three targets on one row.
        let t = targets(&[
            cand(0, 0.0, 0.0, "screen:show:1"),
            cand(1, 250.0, 0.0, "screen:show:2"),
            cand(2, 500.0, 0.0, "screen:show:3"),
        ]);
        let f0 = focus(&t, 0);
        assert_eq!(navigate(&t, Some(&f0), NavDirection::Right), Some(1));
        let f2 = focus(&t, 2);
        assert_eq!(
            navigate(&t, Some(&f2), NavDirection::Right),
            Some(2),
            "clamps"
        );
        assert_eq!(navigate(&t, Some(&f2), NavDirection::Left), Some(1));
    }

    #[test]
    fn straight_ahead_beats_nearer_but_far_off_axis() {
        // From a tab, Down should land on the row list under it, not a nearer
        // target far to the side.
        let t = targets(&[
            cand(0, 200.0, 0.0, "screen:show:1"),
            cand(1, 700.0, 40.0, "side:1"),
            cand(2, 200.0, 90.0, "row:1"),
        ]);
        let f0 = focus(&t, 0);
        assert_eq!(navigate(&t, Some(&f0), NavDirection::Down), Some(2));
    }

    #[test]
    fn vanished_focus_reanchors_from_its_last_rect() {
        let t = targets(&[cand(0, 0.0, 0.0, "a:1"), cand(2, 0.0, 100.0, "a:3")]);
        // The focused region (index 9) is gone; its rect sat between the two.
        let gone = FocusRef {
            index: 9,
            rect: [0.0, 50.0, 200.0, 30.0],
        };
        assert_eq!(navigate(&t, Some(&gone), NavDirection::Down), Some(2));
        assert_eq!(navigate(&t, Some(&gone), NavDirection::Up), Some(0));
    }

    #[test]
    fn empty_targets_focus_nothing() {
        assert_eq!(navigate(&[], None, NavDirection::Down), None);
        let gone = FocusRef {
            index: 0,
            rect: [0.0, 0.0, 10.0, 10.0],
        };
        assert_eq!(navigate(&[], Some(&gone), NavDirection::Down), None);
    }
}

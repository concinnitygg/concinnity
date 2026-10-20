//! The color each glyph of a label draws in: its run's, or the label's own.

use crate::components::ColorRun;

/// Walks a label's color runs alongside its glyphs. Characters are asked for
/// in increasing authored order, so each run is passed over once.
pub(super) struct RunColors<'a> {
    runs: &'a [ColorRun],
    base: [f32; 3],
    next: usize,
}

impl<'a> RunColors<'a> {
    pub(super) fn new(runs: &'a [ColorRun], base: [f32; 3]) -> Self {
        Self {
            runs,
            base,
            next: 0,
        }
    }

    /// The color of authored character `at`.
    pub(super) fn at(&mut self, at: usize) -> [f32; 3] {
        let at = at as u64;
        while self.runs.get(self.next).is_some_and(|r| r.end() <= at) {
            self.next += 1;
        }
        match self.runs.get(self.next) {
            Some(r) if u64::from(r.start) <= at => r.color,
            _ => self.base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const BASE: [f32; 3] = [1.0, 1.0, 1.0];
    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

    fn run(start: u32, length: u32, color: [f32; 3]) -> ColorRun {
        ColorRun {
            start,
            length,
            color,
        }
    }

    #[test]
    fn characters_outside_every_run_take_the_base_color() {
        let runs = [run(2, 2, RED), run(5, 1, BLUE)];
        let mut colors = RunColors::new(&runs, BASE);
        let got: Vec<_> = (0..8).map(|i| colors.at(i)).collect();
        assert_eq!(got, [BASE, BASE, RED, RED, BASE, BLUE, BASE, BASE]);
    }

    #[test]
    fn a_repeated_index_keeps_its_color() {
        let runs = [run(1, 1, RED)];
        let mut colors = RunColors::new(&runs, BASE);
        assert_eq!(colors.at(1), RED);
        assert_eq!(colors.at(1), RED, "a line break reuses the index before it");
        assert_eq!(colors.at(2), BASE);
    }

    #[test]
    fn an_empty_run_colors_nothing() {
        let runs = [run(1, 0, RED), run(1, 1, BLUE)];
        let mut colors = RunColors::new(&runs, BASE);
        assert_eq!(colors.at(0), BASE);
        assert_eq!(colors.at(1), BLUE);
    }
}

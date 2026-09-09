//! The read-out drawn over the corridor: the frame rate, what the device holds,
//! and where the run is.
//!
//! It is on screen for the whole run, so its cost belongs to every segment
//! equally rather than to any one station. What it exercises is the text and
//! sprite path and the two systems that keep those labels current.

use concinnity::components::{FpsCounter, Sprite, StatHud, TextAlign, TextLabel};
use concinnity::cook::{Font, WorldBuilder};

use crate::palette;

// Where the read-out sits, in logical units from the top left, and how far
// apart its rows are.
const ORIGIN: [f32; 2] = [18.0, 16.0];
const ROW_HEIGHT: f32 = 22.0;
const LABEL_SCALE: f32 = 1.0;

// The plate the rows are drawn over.
const PLATE_SIZE: [f32; 2] = [240.0, 168.0];

// The six readings the stat hud fills in, in the order they are stacked.
const READINGS: [&str; 6] = [
    "hud_fps",
    "hud_gpu_wait",
    "hud_vram",
    "hud_ram",
    "hud_ev",
    "hud_edr",
];

/// Declare the read-out.
pub(crate) fn declare(world: &mut WorldBuilder) {
    world.add(
        "hud_font",
        Font {
            size_px: 18,
            ..Default::default()
        },
    );

    world
        .add(
            "hud_plate",
            Sprite {
                x: ORIGIN[0] - 10.0,
                y: ORIGIN[1] - 10.0,
                width: PLATE_SIZE[0],
                height: PLATE_SIZE[1],
                tint: [0.04, 0.05, 0.07, 0.72],
                visible: true,
                ..Default::default()
            },
        )
        .reference("texture", palette::SCUFF);

    for (row, name) in READINGS.iter().enumerate() {
        world.add(*name, reading(row)).reference("font", "hud_font");
    }
    // The counter keeps a row of its own, so the two systems that write frame
    // rate are both on screen rather than one shadowing the other.
    world
        .add("hud_counter_label", reading(READINGS.len()))
        .reference("font", "hud_font");

    world
        .add("hud_stats", StatHud::default())
        .reference("fps_label", READINGS[0])
        .reference("gpu_wait_label", READINGS[1])
        .reference("vram_label", READINGS[2])
        .reference("ram_label", READINGS[3])
        .reference("ev_label", READINGS[4])
        .reference("edr_label", READINGS[5]);

    world
        .add("hud_counter", FpsCounter::default())
        .reference("label", "hud_counter_label");
}

fn reading(row: usize) -> TextLabel {
    TextLabel {
        content: String::new(),
        x: ORIGIN[0],
        y: ORIGIN[1] + row as f32 * ROW_HEIGHT,
        color: [0.86, 0.90, 0.96],
        scale: LABEL_SCALE,
        align: TextAlign::Left,
        visible: true,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The rows have to fit the plate they are drawn over, or the read-out runs
    // off the bottom of its own background.
    #[test]
    fn the_rows_fit_the_plate_behind_them() {
        let rows = READINGS.len() + 1;
        let used = rows as f32 * ROW_HEIGHT;
        assert!(used < PLATE_SIZE[1], "{used} of {}", PLATE_SIZE[1]);
    }

    #[test]
    fn every_reading_has_a_label_of_its_own() {
        let mut names = READINGS.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), READINGS.len());
    }
}

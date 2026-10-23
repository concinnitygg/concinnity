//! The water's wake: the waves bodies make crossing a planet sea, on top of
//! its wind sea and shore waves. A square window of cells around the camera
//! holds the water's height and how fast it rises, and every frame the GPU
//! runs it forward as linear deep-water waves, exactly, whatever the time
//! step: into the spectrum, each wave turned on by its own frequency
//! (`w^2 = g k`), and back. Long waves outrun short ones, so a splash spreads
//! into a train of rings and a moving body draws a Kelvin wake.
//!
//! Bodies push in the change in the water they displace since last frame,
//! spread over a smooth disc a little wider than each piece: a body entering
//! raises the water around it, one leaving lets it fall back, one moving
//! raises it ahead and lets it fall behind. A body riding the waves still
//! pushes nothing, since its pieces are measured against the surface they
//! float on. Stirred and steep water foams, and the foam stays where it formed
//! as it fades.
//!
//! The window follows the camera, led by where it is heading, a whole cell at
//! a time and addressed toroidally (see [`window`]), and damps the waves
//! across a band at its edge so none wrap around. When the simulated frame
//! moves, the next frame reads the field it inherits through the inverse
//! move. Pure CPU; `water_wake.hlsl` mirrors the constants below.

pub mod stirs;
pub mod window;

#[cfg(test)]
mod model;

use crate::planet::Rebase;
use crate::render::environment::grass::{map_at_height, map_then};
use crate::render::uniforms::grass::GRASS_IDENTITY_MAP;
use crate::render::uniforms::wake::{
    MAX_WAKE_STIRS, WAKE_CARRIED, WAKE_LAST_VALID, WAKE_WRITE_SECOND, WaterWakeGpu,
    WaterWakeParams, WaterWakeStirGpu,
};
use crate::water::spectrum::GRAVITY;
use crate::water::stir::WaterStir;
use concinnity_core::math::{SaturatingFrom, exp, sqrt};
use core::cell::Cell;
use stirs::StirMemory;
use window::WakeWindow;

/// Cells along each side of the window. A power of two, so wrapping a cell
/// coordinate is a mask. Mirrors `WATER_WAKE_SIZE` in `water_wake.hlsl`.
pub const WAKE_RESOLUTION: u32 = 256;

/// Edge of one cell, in meters.
pub const WAKE_CELL_SIZE: f32 = 0.25;

/// Width of the band at the window's edge the waves die out across, in
/// meters.
pub const WAKE_SPONGE: f32 = 6.0;

/// How much faster than everywhere else the waves die at the window's very
/// edge, per second.
const WAKE_SPONGE_RATE: f32 = 6.0;

/// The share of its height every wave loses per second.
const WAKE_DAMPING: f32 = 0.08;

/// How much faster short waves die, per second per square radian per meter
/// of wavenumber: a half-meter wave fades in about a second and a half.
const WAKE_VISCOSITY: f32 = 0.004;

/// The longest step the wake takes at once, in seconds, so a stall does not
/// throw a frame's stirs a long way.
const WAKE_MAX_STEP: f32 = 1.0 / 15.0;

/// How wide a piece's push spreads, as a multiple of its radius. Mirrors
/// `WATER_WAKE_SPREAD` in `water_wake.hlsl`.
pub const STIR_SPREAD: f32 = 2.0;

/// The share of the water a body displaces that leaves as waves; the rest
/// only flows around it.
const WAKE_COUPLING: f32 = 1.0;

/// How far a piece may move between frames, in its own radii, before it is
/// taken to have jumped rather than moved.
const WAKE_JUMP_RADII: f32 = 4.0;

/// Seconds of foam life: foam fades to `1/e` over this.
const WAKE_FOAM_SECONDS: f32 = 4.0;

/// Foam per meter of water a stir pushes up or down.
const WAKE_CHURN_FOAM: f32 = 0.4;

/// The wave steepness (rise per meter) past which the wake foams, and the
/// foam per second per unit of steepness past it.
const WAKE_STEEP: f32 = 0.45;
const WAKE_STEEP_FOAM: f32 = 1.0;

/// How high the wake may stand per meter of water under it, the depth
/// waves break at; the rest turns to foam.
pub const WAKE_MAX_DEPTH_RATIO: f32 = 0.78;

/// How far ahead of the camera the window leads, in seconds of its motion,
/// at most `WAKE_MAX_LEAD` meters, and how quickly the lead follows a change
/// in its motion, in seconds.
const WAKE_LEAD_SECONDS: f32 = 1.0;
const WAKE_MAX_LEAD: f32 = 12.0;
const WAKE_LEAD_SMOOTHING: f32 = 0.5;

/// A camera move between frames this far or farther, in meters, is a cut,
/// not motion.
const WAKE_CUT: f32 = 50.0;

/// Threads in one FFT group of the wake: one butterfly per pair of cells.
/// Mirrors `WATER_WAKE_THREADS` in `water_wake.hlsl`.
pub const WAKE_FFT_THREADS: u32 = WAKE_RESOLUTION / 2;

/// The edge of the square group the shading and mip kernels run in. Mirrors
/// `WATER_WAKE_GROUP` in `water_wake.hlsl`.
pub const WAKE_GROUP: u32 = 8;

/// Groups of the rows and inverse kernels, two rows each.
pub const WAKE_ROW_GROUPS: u32 = WAKE_RESOLUTION / 2;

/// Groups of the columns kernel: each column with its mirror, the two that
/// are their own mirrors alone.
pub const WAKE_COLUMN_GROUPS: u32 = WAKE_RESOLUTION / 2 + 1;

/// Mip levels of the field the water draw samples, down to one texel.
pub const WAKE_MIP_LEVELS: u32 = WAKE_RESOLUTION.trailing_zeros() + 1;

/// Groups along each side of a shading or mip kernel writing mip `level`.
pub fn wake_groups(level: u32) -> u32 {
    (WAKE_RESOLUTION >> level).max(1).div_ceil(WAKE_GROUP)
}

/// What the wake carries from frame to frame.
#[derive(Debug, Clone)]
pub struct WakeHistory {
    last: Option<WakeLast>,
    // Rows taking a point in this frame to last frame's `x` and `z`.
    carry: [[f32; 4]; 2],
    carried: bool,
    // The window's lead ahead of the camera, in meters along `x` and `z`.
    lead: [f32; 2],
    // Whether this frame writes the state's second half.
    second: bool,
    // Whether the last frame advanced to was run on the GPU.
    ran: Cell<bool>,
    stirs: StirMemory,
}

#[derive(Debug, Clone, Copy)]
struct WakeLast {
    window: WakeWindow,
    time: f64,
    camera: [f32; 3],
}

impl Default for WakeHistory {
    fn default() -> Self {
        Self {
            last: None,
            carry: GRASS_IDENTITY_MAP,
            carried: false,
            lead: [0.0; 2],
            second: false,
            ran: Cell::new(true),
            stirs: StirMemory::default(),
        }
    }
}

/// One frame of the wake: the simulation's block and the lane the draw
/// reads it through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WakeFrame {
    /// The simulation's block.
    pub params: WaterWakeParams,
    /// Where the sea's draw finds the field.
    pub draw: WaterWakeGpu,
}

impl WakeHistory {
    /// Carry the wake into the frame the world moved to: the next frame reads
    /// the field it inherits through the inverse move, and the bodies' last
    /// pieces move with everything else.
    pub fn rebase(&mut self, rebase: &Rebase) {
        self.carry = map_then(self.carry, &rebase.inverse_matrix());
        self.carried = true;
        if let Some(last) = &mut self.last {
            last.camera = rebase.apply_point(last.camera);
        }
        let lead = rebase.apply_vector([self.lead[0], 0.0, self.lead[1]]);
        self.lead = [lead[0], lead[2]];
        self.stirs.rebase(rebase);
    }

    /// Advance to the frame drawn at `time` seconds, seen from `camera` over
    /// a sea whose surface lies at height `sea` in the simulated frame, with
    /// the floating bodies' `stirs`. Unless [`Self::ran`] marked the last
    /// frame run, its state never reached the GPU and this one starts cold.
    pub fn advance(
        &mut self,
        camera: [f32; 3],
        sea: f32,
        time: f64,
        stirs: &[WaterStir],
    ) -> WakeFrame {
        if !self.ran.replace(false) {
            self.last = None;
        }
        let elapsed = self.last.map_or(0.0, |l| narrow(time - l.time).max(0.0));
        let dt = elapsed.min(WAKE_MAX_STEP);
        self.follow(camera, elapsed);
        let lead = self.lead_offset();
        let window = WakeWindow::around([camera[0] + lead[0], camera[2] + lead[1]]);
        let prev = self.last.map_or(window, |l| l.window);
        let mut flags = 0;
        if self.last.is_some() {
            flags |= WAKE_LAST_VALID;
        }
        if self.carried && self.last.is_some() {
            flags |= WAKE_CARRIED;
        }
        if self.second {
            flags |= WAKE_WRITE_SECOND;
        }
        self.second = !self.second;
        let mut list = [WaterWakeStirGpu::default(); MAX_WAKE_STIRS];
        let count = self.stirs.gather(stirs, &window, camera, &mut list);
        let params = WaterWakeParams {
            window: [
                window.origin[0],
                window.origin[1],
                prev.origin[0],
                prev.origin[1],
            ],
            carry: map_at_height(self.carry, sea),
            grid: [
                WAKE_RESOLUTION,
                WAKE_RESOLUTION.trailing_zeros(),
                u32::saturating_from(count),
                flags,
            ],
            cell: [WAKE_CELL_SIZE, dt, GRAVITY, WAKE_SPONGE],
            damping: [WAKE_DAMPING, WAKE_VISCOSITY, WAKE_SPONGE_RATE, 0.0],
            foam: [
                exp(-dt / WAKE_FOAM_SECONDS),
                WAKE_CHURN_FOAM,
                WAKE_STEEP,
                WAKE_STEEP_FOAM,
            ],
            stirs: list,
        };
        self.last = Some(WakeLast {
            window,
            time,
            camera,
        });
        self.carry = GRASS_IDENTITY_MAP;
        self.carried = false;
        WakeFrame {
            params,
            draw: draw_lane(&window),
        }
    }

    /// Mark the frame last advanced to as run: its kernels were encoded.
    pub fn ran(&self) {
        self.ran.set(true);
    }

    // Follow the camera's motion over `elapsed` seconds into the lead.
    fn follow(&mut self, camera: [f32; 3], elapsed: f32) {
        let Some(last) = self.last.filter(|_| elapsed > 0.0) else {
            return;
        };
        let d = [camera[0] - last.camera[0], camera[2] - last.camera[2]];
        if d[0] * d[0] + d[1] * d[1] >= WAKE_CUT * WAKE_CUT {
            self.lead = [0.0; 2];
            return;
        }
        let k = 1.0 - exp(-elapsed / WAKE_LEAD_SMOOTHING);
        for (lead, step) in self.lead.iter_mut().zip(d) {
            *lead += (step / elapsed - *lead) * k;
        }
    }

    // How far ahead of the camera the window centers, in meters.
    fn lead_offset(&self) -> [f32; 2] {
        let ahead = self.lead.map(|v| v * WAKE_LEAD_SECONDS);
        let length = sqrt(ahead[0] * ahead[0] + ahead[1] * ahead[1]);
        let k = if length > WAKE_MAX_LEAD {
            WAKE_MAX_LEAD / length
        } else {
            1.0
        };
        ahead.map(|v| v * k)
    }
}

/// The draw's lane for a frame whose window is `window`.
pub fn draw_lane(window: &WakeWindow) -> WaterWakeGpu {
    let width = WAKE_RESOLUTION as f32 * WAKE_CELL_SIZE;
    let c = window.center();
    WaterWakeGpu {
        lookup: [1.0 / width, WAKE_CELL_SIZE, c[0], c[1]],
        fade: [
            0.5 * width - WAKE_SPONGE,
            1.0 / WAKE_SPONGE,
            WAKE_MAX_DEPTH_RATIO,
            1.0,
        ],
    }
}

fn narrow(v: f64) -> f32 {
    concinnity_core::math::narrow_f32(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::MAX_FLUID_PIECES;
    use crate::water::stir::StirPiece;

    fn bob(x: f32, volume: f32) -> WaterStir {
        let mut pieces = [StirPiece::default(); MAX_FLUID_PIECES];
        pieces[0] = StirPiece {
            at: [x, 0.0, 0.0],
            radius: 0.6,
            volume,
            ..StirPiece::default()
        };
        WaterStir {
            entity: crate::ecs::Entity::dangling(),
            up: [0.0, 1.0, 0.0],
            velocity: [0.0; 3],
            pieces,
            count: 1,
        }
    }

    // Advance over a sea at height 0, run.
    fn step(h: &mut WakeHistory, camera: [f32; 3], time: f64, stirs: &[WaterStir]) -> WakeFrame {
        let frame = h.advance(camera, 0.0, time, stirs);
        h.ran();
        frame
    }

    // The first frame has nothing to carry and takes no step; the next steps
    // by the time between them, at most the longest step.
    #[test]
    fn steps_follow_the_drawn_time() {
        let mut h = WakeHistory::default();
        let a = step(&mut h, [0.0; 3], 10.0, &[]);
        assert_eq!(a.params.grid[3], 0);
        assert_eq!(a.params.cell[1], 0.0);
        let b = step(&mut h, [0.0; 3], 10.02, &[]);
        assert_eq!(b.params.grid[3], WAKE_LAST_VALID | WAKE_WRITE_SECOND);
        assert!((b.params.cell[1] - 0.02).abs() < 1e-6);
        assert!(b.params.foam[0] < 1.0);
        let c = step(&mut h, [0.0; 3], 11.0, &[]);
        assert_eq!(c.params.cell[1], WAKE_MAX_STEP);
        // A paused world takes no step and keeps all its foam.
        let d = step(&mut h, [0.0; 3], 11.0, &[]);
        assert_eq!(d.params.cell[1], 0.0);
        assert_eq!(d.params.foam[0], 1.0);
        assert_eq!(d.params.grid[..2], [WAKE_RESOLUTION, 8]);
    }

    // The window leads a moving camera, at most `WAKE_MAX_LEAD` ahead, and
    // a cut drops the lead.
    #[test]
    fn the_window_leads_a_moving_camera() {
        let mut h = WakeHistory::default();
        let mut x = 0.0f32;
        let mut frame = step(&mut h, [x, 0.0, 0.0], 0.0, &[]);
        for i in 1..=120 {
            x += 0.5;
            frame = step(&mut h, [x, 0.0, 0.0], f64::from(i) / 60.0, &[]);
        }
        let center = frame.draw.lookup[2];
        assert!(center - x > WAKE_MAX_LEAD - 1.0, "{center} {x}");
        assert!(center - x <= WAKE_MAX_LEAD + WAKE_CELL_SIZE);
        let cut = step(&mut h, [x + 500.0, 0.0, 0.0], 121.0 / 60.0, &[]);
        assert!((cut.draw.lookup[2] - (x + 500.0)).abs() <= WAKE_CELL_SIZE);
    }

    // A frame move hands the next frame the inverse move to read its field
    // through, once, and carries the bodies so a still one pushes nothing.
    #[test]
    fn a_frame_move_is_carried_once() {
        let mut h = WakeHistory::default();
        step(&mut h, [0.0; 3], 0.0, &[bob(2.0, 0.3)]);
        let rebase = Rebase {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [-100.0, 0.0, 0.0],
        };
        h.rebase(&rebase);
        let moved = step(&mut h, [-100.0, 0.0, 0.0], 0.02, &[bob(-98.0, 0.3)]);
        assert_eq!(
            moved.params.grid[3] & !WAKE_WRITE_SECOND,
            WAKE_LAST_VALID | WAKE_CARRIED
        );
        assert_eq!(moved.params.grid[2], 0);
        // A cell at x in this frame was at x + 100 in the last.
        let row = moved.params.carry[0];
        assert!((row[0] * 3.0 + row[2] - 103.0).abs() < 1e-4);
        let next = step(&mut h, [-100.0, 0.0, 0.0], 0.04, &[bob(-97.9, 0.3)]);
        assert_eq!(next.params.grid[3] & !WAKE_WRITE_SECOND, WAKE_LAST_VALID);
        assert_ne!(next.params.grid[3], moved.params.grid[3] & !WAKE_CARRIED);
        assert_eq!(next.params.carry, map_at_height(GRASS_IDENTITY_MAP, 0.0));
        assert_eq!(next.params.grid[2], 1);
    }

    // A frame advanced to but never run leaves the next to start cold, since
    // the half it would read was never written.
    #[test]
    fn a_frame_never_run_starts_the_next_cold() {
        let mut h = WakeHistory::default();
        step(&mut h, [0.0; 3], 0.0, &[]);
        let unrun = h.advance([0.0; 3], 0.0, 0.02, &[]);
        assert_eq!(unrun.params.grid[3] & WAKE_LAST_VALID, WAKE_LAST_VALID);
        let cold = step(&mut h, [0.0; 3], 0.04, &[]);
        assert_eq!(cold.params.grid[3] & WAKE_LAST_VALID, 0);
        assert_eq!(cold.params.cell[1], 0.0);
        let warm = step(&mut h, [0.0; 3], 0.06, &[]);
        assert_eq!(warm.params.grid[3] & WAKE_LAST_VALID, WAKE_LAST_VALID);
    }

    // The carry is read on the sea, wherever the camera is: after a move of
    // a frame tilted by the planet's curve, the field a body stirred is still
    // found under it with the camera high above the water.
    #[test]
    fn the_carry_is_read_on_the_sea() {
        let tilt = 0.02f32;
        let rebase = Rebase {
            rotation: [0.0, 0.0, libm::sinf(0.5 * tilt), libm::cosf(0.5 * tilt)],
            translation: [-100.0, 0.0, 0.0],
        };
        let sea = -2.0;
        let camera = [0.0, 100.0, 0.0];
        let mut h = WakeHistory::default();
        h.advance(camera, sea, 0.0, &[]);
        h.ran();
        h.rebase(&rebase);
        let moved = h.advance(camera, sea, 0.02, &[]);
        // A point on the sea in this frame, carried back, lands where the
        // inverse move puts it in the last.
        let here = [3.0, sea, 1.0];
        let want = rebase.inverse().apply_point(here);
        let row = moved.params.carry;
        let x = row[0][0] * here[0] + row[0][1] * here[2] + row[0][2];
        let z = row[1][0] * here[0] + row[1][1] * here[2] + row[1][2];
        assert!(
            (x - want[0]).abs() < 1e-3 && (z - want[2]).abs() < 1e-3,
            "{x} {z} {want:?}"
        );
        // Read at the camera's height instead, it would miss by meters.
        let off = map_at_height(
            map_then(GRASS_IDENTITY_MAP, &rebase.inverse_matrix()),
            camera[1],
        );
        assert!((off[0][0] * here[0] + off[0][1] * here[2] + off[0][2] - want[0]).abs() > 1.0);
    }

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::WATER_WAKE;
        let uint = |name| crate::render::shader::consts::uint(src, name);
        assert_eq!(uint("WATER_WAKE_SIZE"), WAKE_RESOLUTION as usize);
        assert_eq!(uint("WATER_WAKE_THREADS"), WAKE_FFT_THREADS as usize);
        assert_eq!(uint("WATER_WAKE_MAX_STIRS"), MAX_WAKE_STIRS);
        assert_eq!(uint("WATER_WAKE_GROUP"), WAKE_GROUP as usize);
        assert_eq!(uint("WATER_WAKE_LAST_VALID"), WAKE_LAST_VALID as usize);
        assert_eq!(uint("WATER_WAKE_CARRIED"), WAKE_CARRIED as usize);
        assert_eq!(uint("WATER_WAKE_WRITE_SECOND"), WAKE_WRITE_SECOND as usize);
        let spread = crate::render::shader::consts::float(src, "WATER_WAKE_SPREAD");
        assert_eq!(spread, STIR_SPREAD);
        assert!(src.contains("WaterWakeStir stirs[64];"));
        assert_eq!(WAKE_MIP_LEVELS, 9);
        assert_eq!(wake_groups(0), 32);
        assert_eq!(wake_groups(8), 1);
    }

    #[test]
    fn the_draw_lane_spans_the_window() {
        let lane = draw_lane(&WakeWindow::around([10.0, -4.0]));
        assert_eq!(lane.lookup[0], 1.0 / 64.0);
        assert_eq!(lane.lookup[2], 10.0);
        assert_eq!(lane.lookup[3], -4.0);
        assert_eq!(lane.fade[0], 32.0 - WAKE_SPONGE);
        assert_eq!(lane.fade[3], 1.0);
    }
}

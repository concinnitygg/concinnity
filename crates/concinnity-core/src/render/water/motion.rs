//! A body's waves and ripples at one instant, computed on the host in f64 and
//! wrapped into small ranges, so the shader never multiplies a large time by a
//! frequency, nor reads a phase or a noise coordinate that grew with the
//! surface's distance from the origin. The shader adds only what varies across
//! the surface, measured from an anchor: the surface's center for a summed
//! body, the snapped camera for a simulated ocean (see [`WaterBody::anchor`]),
//! so a sea kilometers across stays precise at its edges too.

use super::ocean::MAX_OCEAN_CASCADES;
use super::ocean::clock::tile_offset;
use super::ripples::{NOISE_PERIOD, RIPPLE_CREST, RIPPLES, WARP_DRIFT, WARP_FREQUENCY};
use super::{MAX_WAVES, WaterBody};
use core::f64::consts::TAU;

/// What moves on a body at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyMotion {
    /// The world-space point everything below is measured from: the anchor's
    /// XZ at the surface's height.
    pub anchor: [f32; 3],
    /// The surface's rectangle relative to the anchor, `[min x, min z, max
    /// x, max z]`.
    pub rest_bounds: [f32; 4],
    /// Each wave's phase at the anchor, in radians in `[0, tau)`; zero past
    /// the body's waves.
    pub wave_phases: [f32; MAX_WAVES],
    /// Each ripple octave's noise offset at the anchor, `[along its heading,
    /// along its crests]`, in lattice cells within half a [`NOISE_PERIOD`] of
    /// zero.
    pub ripple_offsets: [[f32; 2]; RIPPLES],
    /// The ripples' domain warp offset at the anchor, in its noise cells
    /// within half a [`NOISE_PERIOD`] of zero.
    pub warp_offset: [f32; 2],
    /// Where the anchor falls within each cascade's tile, as texture
    /// coordinates in `[0, 1)`; zero past the cascades and for a summed body.
    pub ocean_offsets: [[f32; 2]; MAX_OCEAN_CASCADES],
}

impl WaterBody {
    /// Its waves and ripples `time` seconds into the world, measured from
    /// world XZ `anchor`.
    pub fn motion_at(&self, time: f64, anchor: [f64; 2]) -> BodyMotion {
        let dot = |d: [f32; 2]| f64::from(d[0]) * anchor[0] + f64::from(d[1]) * anchor[1];

        let mut wave_phases = [0.0; MAX_WAVES];
        for (slot, w) in wave_phases.iter_mut().zip(&self.waves) {
            let wave = &w.wave;
            let k = f64::from(wave.wavenumber());
            let omega = f64::from(wave.angular_speed());
            *slot = wrap_phase(f64::from(wave.phase) + k * dot(wave.direction) - omega * time);
        }

        let period = f64::from(NOISE_PERIOD);
        let ripple_offsets = self.ripples.map(|r| {
            let inv_wavelength = 1.0 / f64::from(r.wavelength);
            let seed = f64::from(r.seed);
            let along = (dot(r.direction) - f64::from(r.speed) * time) * inv_wavelength + seed;
            let across = dot(r.across()) * f64::from(RIPPLE_CREST) * inv_wavelength + seed;
            [wrap_centered(along, period), wrap_centered(across, period)]
        });

        let drift = f64::from(WARP_DRIFT) * time;
        let wind = self.wind.direction;
        let warp = |axis: usize| {
            let at = (anchor[axis] - f64::from(wind[axis]) * drift) * f64::from(WARP_FREQUENCY);
            wrap_centered(at, period)
        };
        let s = &self.surface;
        let bound = |axis: usize, side: f32| {
            let c = f64::from(s.center[2 * axis]);
            let e = f64::from(s.extent[axis]);
            ((c + f64::from(side) * e) - anchor[axis]) as f32
        };
        let mut ocean_offsets = [[0.0; 2]; MAX_OCEAN_CASCADES];
        if let Some(ocean) = &self.ocean {
            for (slot, &length) in ocean_offsets.iter_mut().zip(ocean.cascades.lengths()) {
                *slot = tile_offset(anchor, length);
            }
        }
        BodyMotion {
            anchor: [anchor[0] as f32, s.center[1], anchor[1] as f32],
            rest_bounds: [bound(0, -1.0), bound(1, -1.0), bound(0, 1.0), bound(1, 1.0)],
            wave_phases,
            ripple_offsets,
            warp_offset: [warp(0), warp(1)],
            ocean_offsets,
        }
    }
}

// `x` radians wrapped into `[0, tau)`. A remainder that rounds up to tau in
// f32 lands on 0 instead, the same angle.
fn wrap_phase(x: f64) -> f32 {
    let r = wrap(x, TAU) as f32;
    if r >= core::f32::consts::TAU { 0.0 } else { r }
}

// `x` wrapped into `[-period / 2, period / 2)`.
fn wrap_centered(x: f64, period: f64) -> f32 {
    (wrap(x + 0.5 * period, period) - 0.5 * period) as f32
}

// `x` wrapped into `[0, period)`.
fn wrap(x: f64, period: f64) -> f64 {
    let r = libm::fmod(x, period);
    let r = if r < 0.0 { r + period } else { r };
    if r >= period { 0.0 } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::render::wind::WindField;

    fn far_body() -> WaterBody {
        let surface = WaterSurface {
            center: [5321.25, 2.0, -7410.5],
            ..Default::default()
        };
        WaterBody::resolve(&surface, &WindField::new(Some(&Wind::default())))
    }

    fn center_of(body: &WaterBody) -> [f64; 2] {
        body.anchor([0.0; 3])
    }

    // How far `wrapped` sits from `unwrapped` as angles, in radians.
    fn angle_error(wrapped: f32, unwrapped: f64) -> f64 {
        let d = wrap(f64::from(wrapped) - unwrapped, TAU);
        d.min(TAU - d)
    }

    // The unwrapped phase of wave `i` at the surface center, in f64.
    fn unwrapped_phase(body: &WaterBody, i: usize, time: f64) -> f64 {
        let w = &body.waves[i].wave;
        let c = body.surface.center;
        let x = f64::from(w.direction[0]) * f64::from(c[0])
            + f64::from(w.direction[1]) * f64::from(c[2]);
        f64::from(w.phase) + f64::from(w.wavenumber()) * x - f64::from(w.angular_speed()) * time
    }

    // Every phase lies in [0, tau) and is the same angle as the formula it
    // stands in for, just after the start and after twelve days of uptime.
    #[test]
    fn wave_phases_wrap_and_match_the_unwrapped_formula() {
        let body = far_body();
        assert!(!body.waves.is_empty());
        for time in [0.0, 0.25, 1.0e6 + 0.125] {
            let motion = body.motion_at(time, center_of(&body));
            for (i, &phase) in motion.wave_phases.iter().enumerate() {
                assert!(
                    (0.0..core::f32::consts::TAU).contains(&phase),
                    "{i} {phase}"
                );
                if i < body.waves.len() {
                    let err = angle_error(phase, unwrapped_phase(&body, i, time));
                    assert!(err < 1e-5, "wave {i} at {time}: {err}");
                } else {
                    assert_eq!(phase, 0.0);
                }
            }
        }
    }

    // The ripple and warp offsets stay within half a period, and are the
    // unwrapped offsets modulo it.
    #[test]
    fn ripple_offsets_wrap_and_match_the_unwrapped_formula() {
        let body = far_body();
        let c = body.surface.center;
        let center = [f64::from(c[0]), f64::from(c[2])];
        let period = f64::from(NOISE_PERIOD);
        let half = 0.5 * NOISE_PERIOD;
        let same_cell = |wrapped: f32, unwrapped: f64| {
            let d = wrap(f64::from(wrapped) - unwrapped, period);
            d.min(period - d)
        };
        for time in [0.0, 0.25, 1.0e6 + 0.125] {
            let motion = body.motion_at(time, center_of(&body));
            for (r, ripple) in body.ripples.iter().enumerate() {
                let [along, across] = motion.ripple_offsets[r];
                assert!((-half..half).contains(&along) && (-half..half).contains(&across));
                let dot = |d: [f32; 2]| f64::from(d[0]) * center[0] + f64::from(d[1]) * center[1];
                let wl = f64::from(ripple.wavelength);
                let seed = f64::from(ripple.seed);
                let want_along =
                    (dot(ripple.direction) - f64::from(ripple.speed) * time) / wl + seed;
                let want_across = dot(ripple.across()) * f64::from(RIPPLE_CREST) / wl + seed;
                assert!(same_cell(along, want_along) < 1e-3, "{r} {time}");
                assert!(same_cell(across, want_across) < 1e-3, "{r} {time}");
            }
            for (axis, &w) in motion.warp_offset.iter().enumerate() {
                assert!((-half..half).contains(&w));
                let drift = f64::from(WARP_DRIFT) * time;
                let want = (center[axis] - f64::from(body.wind.direction[axis]) * drift)
                    * f64::from(WARP_FREQUENCY);
                assert!(same_cell(w, want) < 1e-3, "{axis} {time}");
            }
        }
    }

    // A surface at the origin, at the start, carries its own phases and the
    // octave seeds unchanged: nothing near the origin moves.
    #[test]
    fn at_the_origin_at_the_start_nothing_shifts() {
        let body = WaterBody::resolve(
            &WaterSurface::default(),
            &WindField::new(Some(&Wind::default())),
        );
        assert_eq!(body.surface.center, [0.0; 3]);
        let motion = body.motion_at(0.0, center_of(&body));
        for (i, w) in body.waves.iter().enumerate() {
            assert!((motion.wave_phases[i] - w.wave.phase).abs() < 1e-6, "{i}");
        }
        for (r, ripple) in body.ripples.iter().enumerate() {
            assert_eq!(motion.ripple_offsets[r], [ripple.seed, ripple.seed]);
        }
        assert_eq!(motion.warp_offset, [0.0, 0.0]);
        assert_eq!(motion.anchor, [0.0; 3]);
        let [ex, ez] = body.surface.extent;
        assert_eq!(motion.rest_bounds, [-ex, -ez, ex, ez]);
        assert_eq!(motion.ocean_offsets, [[0.0; 2]; MAX_OCEAN_CASCADES]);
    }

    // A summed body far from the origin is measured from its center: its
    // rectangle is its extent around the anchor, exactly.
    #[test]
    fn a_summed_body_is_anchored_at_its_center() {
        let body = far_body();
        let anchor = body.anchor([9.0e5, 0.0, 1.0]);
        assert_eq!(anchor, [5321.25, -7410.5]);
        let m = body.motion_at(3.0, anchor);
        assert_eq!(m.anchor, body.surface.center);
        let [ex, ez] = body.surface.extent;
        assert_eq!(m.rest_bounds, [-ex, -ez, ex, ez]);
    }

    // An ocean is measured from the camera, snapped to its grid, wherever
    // the camera roams; its rectangle and tile offsets follow, and the wave
    // phases at the anchor still match the unwrapped formula there.
    #[test]
    fn an_ocean_is_anchored_at_the_snapped_camera() {
        use super::super::ocean::OceanQuality;
        use super::super::ocean::clock::tile_offset;
        let surface = WaterSurface {
            center: [4.0e5, 0.0, -2.5e5],
            extent: [6000.0, 6000.0],
            fetch_meters: 1.0e5,
            waves: alloc::vec![crate::components::WaterWave::default()],
            ..Default::default()
        };
        let wind = WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }));
        let (bodies, _) = WaterBody::resolve_all([&surface], &wind, OceanQuality::FULL, None);
        let body = &bodies[0];
        let ocean = body.ocean.as_ref().expect("an ocean");
        let camera = [4.0e5 + 5900.25, 12.0, -2.5e5 - 5800.5];
        let anchor = body.anchor(camera);
        assert_eq!(
            anchor,
            ocean
                .grid
                .snap([f64::from(camera[0]), f64::from(camera[2])])
        );
        let m = body.motion_at(1.0e6 + 0.5, anchor);
        assert_eq!(m.anchor[1], 0.0);
        // The surface's far edge is near the camera.
        assert!(
            m.rest_bounds[2] < 100.0 && m.rest_bounds[2] > 0.0,
            "{:?}",
            m.rest_bounds
        );
        assert!(m.rest_bounds[0] < -11000.0);
        for (c, &length) in ocean.cascades.lengths().iter().enumerate() {
            assert_eq!(m.ocean_offsets[c], tile_offset(anchor, length));
        }
        let w = &body.waves[0].wave;
        let x = f64::from(w.direction[0]) * anchor[0] + f64::from(w.direction[1]) * anchor[1];
        let want = f64::from(w.phase) + f64::from(w.wavenumber()) * x
            - f64::from(w.angular_speed()) * (1.0e6 + 0.5);
        assert!(angle_error(m.wave_phases[0], want) < 1e-5);
    }

    #[test]
    fn wrapping_lands_in_range() {
        assert_eq!(wrap(-0.5, 4.0), 3.5);
        assert_eq!(wrap(9.0, 4.0), 1.0);
        assert_eq!(wrap_centered(3.0, 4.0), -1.0);
        assert_eq!(wrap_centered(-2.0, 4.0), -2.0);
        assert_eq!(wrap_centered(2.0, 4.0), -2.0);
        assert_eq!(wrap_phase(-1e-18), 0.0);
    }
}

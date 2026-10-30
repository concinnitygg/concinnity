//! A frame's cloud inputs: where in the noise each layer's sky sits around the
//! camera, held in double precision on the CPU so the GPU reads the noise
//! relative to the camera at any distance from the world origin.

use super::settings::{CloudSettings, MAX_CLOUD_LAYERS};
use super::shadow::CloudShadowSquare;
use crate::gfx::render_types::CloudShadowLanes;
use crate::render::uniforms::DaylightBlock;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::{narrow_f32, round_f64};

/// How far the base shape and the detail move through the noise as a cloud
/// changes into another one, in tiles: about one puff.
const EVOLVE_TILES: f64 = 0.25;
// The direction they move in, so the clouds boil upward as they change.
const EVOLVE_DIR: DVec3 = [0.3, 0.9, 0.3];
// The weather's second, slower field, which moves the clouds' cells against
// the first: its scale against the first's and its share of the drift.
pub(super) const WEATHER_SECOND_SCALE: f64 = 0.61;
const WEATHER_SECOND_DRIFT: f64 = 0.8;

/// Where the camera stands toward the clouds this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudView {
    /// Rows of the rotation from the frame's axes into the planet's fixed
    /// ones (a flat world's are its own).
    pub to_fixed: [DVec3; 3],
    /// The camera relative to the planet's center, in meters, in the frame's
    /// axes.
    pub camera_m: DVec3,
    /// The camera's position in the frame.
    pub camera_pos: [f32; 3],
    /// The clouds' clock, in seconds.
    pub time: f64,
}

impl CloudView {
    // `v` in the planet's fixed axes.
    fn fixed(&self, v: DVec3) -> DVec3 {
        self.to_fixed.map(|row| dvec3::dot(row, v))
    }
}

/// One layer's noise origins this frame: the noise coordinates, in tiles and
/// wrapped into one tile, that the camera (or for the weather, the point of
/// the layer's bottom straight above or below it) reads.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CloudLayerFrame {
    /// The weather's first field.
    pub weather: [f32; 3],
    /// The weather's second field.
    pub weather_second: [f32; 3],
    /// The base shape.
    pub shape: [f32; 3],
    /// The detail.
    pub detail: [f32; 3],
}

/// This frame's cloud inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudFrame {
    /// The layers' origins; the first `count` are drawn.
    pub layers: [CloudLayerFrame; MAX_CLOUD_LAYERS],
    /// Rows of the rotation into the planet's fixed axes.
    pub to_fixed: [[f32; 3]; 3],
    /// The camera relative to the planet's center, in kilometers, in its
    /// fixed axes.
    pub camera_km: [f32; 3],
    /// The light the clouds are lit by: the daylight curves and the bodies,
    /// their directions in the fixed axes.
    pub daylight: DaylightBlock,
    /// Where the shadow map lies, in the fixed axes.
    pub shadow: CloudShadowSquare,
    /// Where the shadow map lies, as the shaded passes read it.
    pub shadow_lanes: CloudShadowLanes,
    /// The clouds' clock, in seconds.
    pub time: f32,
    /// The radius, in kilometers, of the ground the clouds' horizon is
    /// measured from: the one the sky is drawn over, so they catch the light
    /// as long after sunset as the sky does.
    pub horizon_km: f32,
}

// `x` wrapped into [0, 1).
fn wrap(x: f64) -> f64 {
    let r = round_f64(x);
    let f = x - r;
    if f < 0.0 { f + 1.0 } else { f }
}

fn wrap3(v: DVec3) -> [f32; 3] {
    v.map(|c| narrow_f32(wrap(c)))
}

impl CloudFrame {
    /// The inputs for `settings` over a ground `ground_km` from the planet's
    /// center, seen from `view`, lit by `daylight` (in the frame's axes) under
    /// a sky drawn over a ground `horizon_km` in radius.
    pub fn new(
        settings: &CloudSettings,
        ground_km: f32,
        view: &CloudView,
        daylight: &DaylightBlock,
        horizon_km: f32,
    ) -> Self {
        let camera = view.fixed(view.camera_m);
        let up = dvec3::normalize_or(camera, [0.0, 1.0, 0.0]);
        let mut layers = [CloudLayerFrame::default(); MAX_CLOUD_LAYERS];
        for (frame, layer) in layers.iter_mut().zip(settings.layers()) {
            let km = |v: f32| f64::from(v) * 1000.0;
            let [dx, dz] = layer.drift_km_s.map(|d| km(d) * view.time);
            let drift = [dx, 0.0, dz];
            let seed = dvec3::from_f32(layer.seed_offset);
            let evolve = dvec3::scale(
                EVOLVE_DIR,
                EVOLVE_TILES * f64::from(layer.evolve_rate) * view.time,
            );
            let below = layer.stretched(dvec3::scale(up, km(ground_km + layer.base_km)));
            let drift = layer.stretched(drift);
            let tiles = |p: DVec3, tile_km: f32| dvec3::scale(p, 1.0 / km(tile_km));
            frame.weather = wrap3(dvec3::add(
                tiles(dvec3::add(below, drift), layer.weather_tile_km),
                seed,
            ));
            frame.weather_second = wrap3(dvec3::add(
                tiles(
                    dvec3::add(below, dvec3::scale(drift, WEATHER_SECOND_DRIFT)),
                    layer.weather_tile_km * narrow_f32(WEATHER_SECOND_SCALE),
                ),
                dvec3::scale(seed, 1.7),
            ));
            let moved = dvec3::add(layer.stretched(camera), drift);
            frame.shape = wrap3(dvec3::add(
                dvec3::add(tiles(moved, layer.shape_tile_km), seed),
                evolve,
            ));
            frame.detail = wrap3(dvec3::add(
                dvec3::add(tiles(moved, layer.detail_tile_km), seed),
                dvec3::scale(evolve, 2.0),
            ));
        }
        let to_fixed = view.to_fixed.map(dvec3::to_f32);
        let [lo, hi] = settings.span_km();
        let camera_km = dvec3::scale(camera, 1e-3);
        let shadow = CloudShadowSquare::around(
            camera_km,
            f64::from(ground_km),
            f64::from(ground_km + 0.5 * (lo + hi)),
        );
        let [lx, ly, lz, lit] = daylight.body_dir[0];
        let shadow_lanes = if lit > 0.5 {
            shadow.lanes(&view.to_fixed, view.camera_pos, camera_km, [lx, ly, lz])
        } else {
            CloudShadowLanes::OFF
        };
        let mut light = *daylight;
        for dir in &mut light.body_dir {
            let d = view.fixed(dvec3::from_f32([dir[0], dir[1], dir[2]]));
            let [x, y, z] = dvec3::to_f32(d);
            *dir = [x, y, z, dir[3]];
        }
        Self {
            layers,
            to_fixed,
            camera_km: dvec3::to_f32(camera_km),
            daylight: light,
            shadow,
            shadow_lanes,
            time: narrow_f32(view.time),
            horizon_km,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Cloud;
    use crate::render::environment::clouds::CloudMarch;
    use crate::render::uniforms::{DaylightBody, DaylightView};
    use crate::wind::WindField;
    use concinnity_core::approx::close;

    const IDENTITY: [DVec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // A quarter turn about z: the frame's +x is the planet's +y.
    const TURNED: [DVec3; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    fn settings(wind: f32) -> CloudSettings {
        let wind = WindField {
            direction: [1.0, 0.0],
            strength: wind,
            ..WindField::STILL
        };
        CloudSettings::resolve(&[Cloud::default()], &wind, CloudMarch::FULL).unwrap()
    }

    fn view(to_fixed: [DVec3; 3], camera_m: DVec3, time: f64) -> CloudView {
        CloudView {
            to_fixed,
            camera_m,
            camera_pos: [0.0; 3],
            time,
        }
    }

    fn lit() -> DaylightBlock {
        DaylightBlock::new(
            [[0.0; 4]; 96],
            &DaylightView {
                center: [0.0; 3],
                camera_up: [0.0, 1.0, 0.0],
                camera_reach: 0.0,
                reach: [1.0, 1.0],
                ground_radius: 50_000.0,
                ground_height: 5000.0,
                bodies: [
                    Some(DaylightBody {
                        direction: [1.0, 0.0, 0.0],
                        light: [1.0; 3],
                    }),
                    None,
                ],
            },
        )
    }

    // The same point in the world reads the same noise whichever frame it is
    // seen from: a frame move turns the axes and the camera with them.
    #[test]
    fn the_origins_follow_the_world_not_the_frame() {
        let s = settings(0.0);
        let a = CloudFrame::new(
            &s,
            50.0,
            &view(IDENTITY, [0.0, 51_000.0, 0.0], 3.0),
            &lit(),
            50.0,
        );
        let b = CloudFrame::new(
            &s,
            50.0,
            &view(TURNED, [51_000.0, 0.0, 0.0], 3.0),
            &lit(),
            50.0,
        );
        assert!(close(a.layers[0].shape, b.layers[0].shape, 1e-6));
        assert!(close(a.layers[0].weather, b.layers[0].weather, 1e-6));
        assert!(close(a.camera_km, b.camera_km, 1e-4));
        assert!(close(b.camera_km, [0.0, 51.0, 0.0], 1e-4));
        // The sun turns into the planet's axes with the frame.
        let sun = b.daylight.body_dir[0];
        assert!(close([sun[0], sun[1], sun[2]], [0.0, 1.0, 0.0], 1e-6));
    }

    // Moving the camera moves the origins by the distance in tiles.
    #[test]
    fn the_shape_origin_moves_with_the_camera() {
        let s = settings(0.0);
        let tile = f64::from(s.layers()[0].shape_tile_km) * 1000.0;
        let step = 0.1 * tile;
        let a = CloudFrame::new(
            &s,
            50.0,
            &view(IDENTITY, [0.0, 51_000.0, 0.0], 0.0),
            &lit(),
            50.0,
        );
        let b = CloudFrame::new(
            &s,
            50.0,
            &view(IDENTITY, [step, 51_000.0, 0.0], 0.0),
            &lit(),
            50.0,
        );
        let moved = wrap(f64::from(b.layers[0].shape[0] - a.layers[0].shape[0]));
        assert!(close(moved, 0.1, 1e-5), "{moved}");
    }

    // Far from the origin the wrapped origins stay as precise as near it.
    #[test]
    fn far_cameras_keep_their_precision() {
        let s = settings(0.0);
        let far = 6_360_000.0;
        let tile = f64::from(s.layers()[0].detail_tile_km) * 1000.0;
        let a = CloudFrame::new(
            &s,
            6360.0,
            &view(IDENTITY, [0.0, far, 0.0], 0.0),
            &lit(),
            50.0,
        );
        let b = CloudFrame::new(
            &s,
            6360.0,
            &view(IDENTITY, [0.01 * tile, far, 0.0], 0.0),
            &lit(),
            50.0,
        );
        let moved = wrap(f64::from(b.layers[0].detail[0] - a.layers[0].detail[0]));
        assert!(close(moved, 0.01, 1e-4), "{moved}");
    }

    #[test]
    fn the_wind_drifts_the_weather_over_time() {
        let s = settings(10.0);
        let at = |t| {
            CloudFrame::new(
                &s,
                50.0,
                &view(IDENTITY, [0.0, 51_000.0, 0.0], t),
                &lit(),
                50.0,
            )
        };
        let tile = f64::from(s.layers()[0].weather_tile_km) * 1000.0;
        let drift = f64::from(s.layers()[0].drift_km_s[0]) * 1000.0;
        let moved = wrap(f64::from(
            at(10.0).layers[0].weather[0] - at(0.0).layers[0].weather[0],
        ));
        assert!(close(moved, wrap(drift * 10.0 / tile), 1e-5));
        assert!(close(
            at(10.0).layers[0].weather[2],
            at(0.0).layers[0].weather[2],
            1e-6
        ));
    }

    #[test]
    fn origins_wrap_into_one_tile() {
        assert!(close(wrap(3.25), 0.25, 1e-12));
        assert!(close(wrap(-0.25), 0.75, 1e-12));
        assert!(close(wrap(-3.75), 0.25, 1e-12));
    }
}

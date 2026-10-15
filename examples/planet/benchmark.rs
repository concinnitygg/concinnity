//! The benchmark run: the meadow crossed at a run just over the grass, then
//! climbed out of to look along the curve to the horizon, timed frame by frame
//! and reported when the camera reaches the end of its path.
//!
//! A planet is simulated in a frame that moves with the camera, every
//! kilometer, and draws its ground as tiles that refine and coarsen as the
//! camera passes, so its cost is paid along a path rather than at a station.
//! The run crosses several of those moves on the ground, where the grass and
//! the finest tiles are, and more of them aloft, where the tiles reach out to
//! the horizon and the grass gives way to the color it shows from afar.
//!
//! The path is laid out by latitude along the prime meridian, northward from
//! where the world starts, each leg ending a fixed height over the ground the
//! planet's own shape puts there: the camera rides the hills rather than
//! cutting through them. The track runs on the fixed simulation clock, so a
//! slower host visits the same poses at the same track times and simply
//! samples fewer of them.
//!
//! The run can also save a few of its frames, at fixed moments of the path:
//! the meadow by day and by night, and the view from aloft by day. The sky
//! turns from afternoon through night to the next morning over the run, so
//! each moment is placed where the sun is up or down, not only by stretch.

use std::path::Path as FilePath;

use concinnity::components::{
    CameraTravel, CameraTurn, Ease, FrameCapture, FrameReport, GraphicsConfig, Planet,
};
use concinnity::cook::{Camera3D, CameraTrack, WorldBuilder};

// The ground run: this far along the ground, this high over the grass, this
// fast. Long enough to cross three frame moves.
const GROUND_METERS: f64 = 3_200.0;
const GROUND_HEIGHT: f64 = 2.5;
const GROUND_SPEED: f32 = 30.0;
// The climb: up to the cruising height over this much ground.
const CLIMB_METERS: f64 = 1_000.0;
const CLIMB_SPEED: f32 = 40.0;
// The cruise: this high over the ground, this far, this fast.
const ALOFT_METERS: f64 = 3_000.0;
const ALOFT_HEIGHT: f64 = 400.0;
const ALOFT_SPEED: f32 = 75.0;
// How far apart the points the path is laid through are: close enough that a
// leg never strays from the ground under it by more than a few centimeters.
const GROUND_STEP: f64 = 25.0;
const ALOFT_STEP: f64 = 100.0;

const SETTLE_SECONDS: f32 = 4.0;
const WARMUP_SECONDS: f32 = 2.5;
const GROUND_PITCH_DEG: f32 = -6.0;
const ALOFT_PITCH_DEG: f32 = -14.0;
const CAMERA_FOV_Y_DEGREES: f32 = 70.0;

// What the world file declares that the run replaces: the camera, which the
// track drives instead, and the graphics settings.
const CAMERA: &str = "cam";
const GRAPHICS: &str = "GraphicsConfig#0";

// The frames saved with `--capture`: the file name, the segment the frame is
// taken in, and how far through that segment. The sun sets about a third of
// the way along the ground and rises again near the end of the cruise.
const CAPTURES: [(&str, &str, f32); 3] = [
    ("ground-day.png", "ground", 0.15),
    ("aloft-day.png", "aloft", 0.9),
    ("ground-night.png", "ground", 0.9),
];

/// Swap the world's camera for one on the path over `planet`, and time the run
/// uncapped: a capped GPU idles between frames and lowers its clocks. With a
/// `capture_dir`, the run also saves its fixed moments there.
pub(crate) fn inject(
    world: &mut WorldBuilder,
    planet: &Planet,
    graphics: Option<GraphicsConfig>,
    capture_dir: Option<&FilePath>,
) {
    let path = Path::new(planet);
    let captures = capture_dir.map_or_else(Vec::new, |dir| captures(&path.travel, dir));
    world.remove(CAMERA);
    if graphics.is_some() {
        world.remove(GRAPHICS);
    }
    let mut graphics = graphics.unwrap_or_default();
    graphics.frame_pacing.fps_cap = Some(0);
    world.add("graphics", graphics);
    world.add(
        CAMERA,
        Camera3D {
            fov_y_degrees: CAMERA_FOV_Y_DEGREES,
            near: 0.1,
            view_distance: None,
            position: path.start,
            yaw: 0.0,
            pitch: GROUND_PITCH_DEG.to_radians(),
            controller: None,
        },
    );
    world.add(
        "track",
        CameraTrack {
            travel: path.travel,
            turn: path.turn,
        },
    );
    world.add(
        "report",
        FrameReport {
            warmup_seconds: WARMUP_SECONDS,
            captures,
            ..Default::default()
        },
    );
}

// The fixed moments as captures written into `dir`.
fn captures(travel: &[CameraTravel], dir: &FilePath) -> Vec<FrameCapture> {
    CAPTURES
        .iter()
        .filter_map(|&(file, segment, through)| {
            let (start, end) = segment_span(travel, segment)?;
            Some(FrameCapture {
                seconds: start + (end - start) * through,
                path: dir.join(file).to_string_lossy().into_owned(),
            })
        })
        .collect()
}

// When the named segment starts and ends on the track: from the leg that
// opens it to the next leg that opens another.
fn segment_span(travel: &[CameraTravel], segment: &str) -> Option<(f32, f32)> {
    let mut at = 0.0;
    let mut span: Option<(f32, f32)> = None;
    for leg in travel {
        if !leg.segment.is_empty() && span.is_some() {
            break;
        }
        let seconds = leg_seconds(leg);
        if leg.segment == segment {
            span = Some((at, at));
        }
        at += seconds;
        if let Some((_, end)) = &mut span {
            *end = at;
        }
    }
    span
}

// How long a leg runs: its own duration, or the one its distance and speed
// imply.
fn leg_seconds(leg: &CameraTravel) -> f32 {
    if leg.seconds > 0.0 {
        leg.seconds
    } else if leg.speed > 0.0 {
        leg.distance / leg.speed
    } else {
        0.0
    }
}

// The path north from the start: where the camera begins, and the legs it
// travels and turns through.
struct Path {
    start: [f32; 3],
    travel: Vec<CameraTravel>,
    turn: Vec<CameraTurn>,
}

impl Path {
    fn new(planet: &Planet) -> Self {
        let shape = planet.shape();
        // Degrees of latitude per meter along the ground.
        let per_meter = (1.0 / f64::from(planet.radius)).to_degrees();
        let at = |meters: f64, height: f64| shape.geo_point(meters * per_meter, 0.0, height);

        let origin = at(0.0, GROUND_HEIGHT);
        let mut travel = vec![CameraTravel {
            seconds: SETTLE_SECONDS,
            segment: "settle".to_string(),
            ..Default::default()
        }];
        let mut turn = vec![CameraTurn {
            seconds: SETTLE_SECONDS,
            ..Default::default()
        }];
        let mut from = origin;
        for (meters, height, speed, segment) in stops() {
            let to = at(meters, height);
            let step: [f64; 3] = std::array::from_fn(|i| to[i] - from[i]);
            let distance = step.iter().map(|c| c * c).sum::<f64>().sqrt();
            from = to;
            travel.push(CameraTravel {
                direction: step.map(|c| c as f32),
                distance: distance as f32,
                speed,
                ease: Ease::Linear,
                segment: segment.to_string(),
                ..Default::default()
            });
            let lift = ((height - GROUND_HEIGHT) / (ALOFT_HEIGHT - GROUND_HEIGHT)) as f32;
            turn.push(CameraTurn {
                yaw_deg: Some(0.0),
                pitch_deg: Some(GROUND_PITCH_DEG + (ALOFT_PITCH_DEG - GROUND_PITCH_DEG) * lift),
                seconds: distance as f32 / speed,
                ease: Ease::Linear,
                ..Default::default()
            });
        }
        Self {
            start: origin.map(|c| c as f32),
            travel,
            turn,
        }
    }
}

// The points the path is laid through, as meters along the ground, height
// over it, speed to reach them, and the segment a leg opens (empty where it
// continues the one before).
fn stops() -> Vec<(f64, f64, f32, &'static str)> {
    let mut stops = Vec::new();
    let mut along = 0.0;
    let mut first = true;
    while along < GROUND_METERS {
        along = (along + GROUND_STEP).min(GROUND_METERS);
        stops.push((
            along,
            GROUND_HEIGHT,
            GROUND_SPEED,
            if first { "ground" } else { "" },
        ));
        first = false;
    }
    let climb_from = along;
    first = true;
    while along < climb_from + CLIMB_METERS {
        along = (along + ALOFT_STEP).min(climb_from + CLIMB_METERS);
        let t = (along - climb_from) / CLIMB_METERS;
        let eased = t * t * (3.0 - 2.0 * t);
        let height = GROUND_HEIGHT + (ALOFT_HEIGHT - GROUND_HEIGHT) * eased;
        stops.push((along, height, CLIMB_SPEED, if first { "climb" } else { "" }));
        first = false;
    }
    let cruise_from = along;
    first = true;
    while along < cruise_from + ALOFT_METERS {
        along = (along + ALOFT_STEP).min(cruise_from + ALOFT_METERS);
        stops.push((
            along,
            ALOFT_HEIGHT,
            ALOFT_SPEED,
            if first { "aloft" } else { "" },
        ));
        first = false;
    }
    stops
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity::components::{DirectionalLight, SkyRotation};

    fn planet() -> Planet {
        crate::declared_planet(crate::WORLD).expect("the world declares a planet")
    }

    fn sky() -> SkyRotation {
        let args = crate::declared(crate::WORLD, "SkyRotation", &[]).expect("a turning sky");
        serde_json::from_value(args).expect("parses")
    }

    fn sun() -> DirectionalLight {
        let args = crate::declared(crate::WORLD, "DirectionalLight", &[]).expect("a sun");
        serde_json::from_value(args).expect("parses")
    }

    fn moment(file: &str) -> f32 {
        let path = Path::new(&planet());
        captures(&path.travel, FilePath::new("shots"))
            .into_iter()
            .find(|c| c.path.ends_with(file))
            .map(|c| c.seconds)
            .expect("a capture of that name")
    }

    // Where the camera is `t` seconds into the track: each leg runs at an
    // even rate along its step.
    fn camera_at(path: &Path, t: f32) -> [f64; 3] {
        let mut position = path.start.map(f64::from);
        let mut at = 0.0;
        for leg in &path.travel {
            let seconds = leg_seconds(leg);
            let through = if seconds > 0.0 {
                ((t - at) / seconds).clamp(0.0, 1.0)
            } else {
                1.0
            };
            for (p, d) in position.iter_mut().zip(leg.direction) {
                *p += f64::from(d) * f64::from(through);
            }
            at += seconds;
        }
        position
    }

    fn normalize(v: [f64; 3]) -> [f64; 3] {
        let length = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        v.map(|c| c / length)
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    // The sun's height over the camera's horizon `t` seconds into the run, in
    // degrees: the declared light turned with the sky, against the planet's up
    // where the camera is.
    fn sun_elevation_deg(t: f32) -> f64 {
        let planet = planet();
        let sky = sky();
        let light = sun();

        // The sky turns a body the way a planet's spin does: a negative turn
        // about the pole in the right-handed sense.
        let k = normalize(sky.axis.map(f64::from));
        let v = normalize(light.direction.map(f64::from));
        let angle = -f64::from(sky.angle_deg + sky.degrees_per_second * t).to_radians();
        let cross = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        let along = dot(k, v) * (1.0 - angle.cos());
        let sun: [f64; 3] =
            std::array::from_fn(|i| v[i] * angle.cos() + cross[i] * angle.sin() + k[i] * along);

        let camera = camera_at(&Path::new(&planet), t);
        let up = normalize(std::array::from_fn(|i| {
            camera[i] - f64::from(planet.center[i])
        }));
        dot(sun, up).asin().to_degrees()
    }

    #[test]
    fn the_ground_segment_starts_after_the_settle_and_runs_its_length() {
        let path = Path::new(&planet());
        let (start, end) = segment_span(&path.travel, "ground").expect("a ground segment");
        assert_eq!(start, SETTLE_SECONDS);
        let flat = (GROUND_METERS / f64::from(GROUND_SPEED)) as f32;
        // The hills make the run a little longer than the flat distance.
        assert!(
            end - start >= flat * 0.999 && end - start < flat * 1.02,
            "{start}..{end}"
        );
    }

    #[test]
    fn segments_follow_one_another() {
        let path = Path::new(&planet());
        let span = |name| segment_span(&path.travel, name).expect(name);
        assert_eq!(span("ground").1, span("climb").0);
        assert_eq!(span("climb").1, span("aloft").0);
        assert!(segment_span(&path.travel, "nowhere").is_none());
    }

    #[test]
    fn every_capture_lands_in_its_segment_inside_the_directory() {
        let path = Path::new(&planet());
        let dir = FilePath::new("shots");
        let captures = captures(&path.travel, dir);
        assert_eq!(captures.len(), CAPTURES.len());
        for (capture, (file, segment, _)) in captures.iter().zip(CAPTURES) {
            assert_eq!(FilePath::new(&capture.path), dir.join(file));
            let (start, end) = segment_span(&path.travel, segment).expect(segment);
            assert!(
                capture.seconds > start && capture.seconds < end,
                "{capture:?}"
            );
            assert!(capture.seconds > WARMUP_SECONDS);
        }
    }

    #[test]
    fn the_day_captures_have_the_sun_up_and_the_night_one_has_it_down() {
        let ground_day = sun_elevation_deg(moment("ground-day.png"));
        let aloft_day = sun_elevation_deg(moment("aloft-day.png"));
        let ground_night = sun_elevation_deg(moment("ground-night.png"));
        assert!(ground_day > 15.0, "{ground_day} deg");
        assert!(aloft_day > 10.0, "{aloft_day} deg");
        assert!(ground_night < -30.0, "{ground_night} deg");
    }
}

//! The facts a frame's view measured, as passes and reports read them.

use crate::gfx::render_types::MAX_LEVEL_LAYERS;
use concinnity_core::math::asin;

/// How many subjects the facts keep: the planet, each water surface, and the
/// prop a reflection probe rides.
pub const VIEW_SUBJECTS: usize = 2 + MAX_LEVEL_LAYERS;

/// What a subject is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubjectKind {
    /// A planet: its ground and everything standing on it below the height
    /// its ground stays under.
    Planet,
    /// A water surface: a planet's sea or a flat world's water rectangle.
    Water,
    /// The prop a reflection probe rides.
    Prop,
}

impl SubjectKind {
    /// The kind's name, as reports print it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Planet => "planet",
            Self::Water => "water",
            Self::Prop => "prop",
        }
    }
}

/// What a measurement saw of one subject, or of the surfaces no subject
/// holds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sighting {
    /// The share of the screen it covered, in `[0, 1]`.
    pub coverage: f32,
    /// How far its nearest visible part lay from the camera, in meters;
    /// `None` when none of it was seen.
    pub nearest: Option<f32>,
}

/// One large thing in the world as the camera sees it this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    /// What it is.
    pub kind: SubjectKind,
    /// How large it is, in meters: a sphere's diameter, or the diagonal of a
    /// box or rectangle.
    pub size: f32,
    /// How far its nearest point lies from the camera, in meters, seen or
    /// not; 0 for a camera inside it. For a planet this is the camera's
    /// height over its ground, for a sea the height over its surface.
    pub distance: f32,
    /// How far its center lies from the camera, in meters.
    pub center_distance: f32,
    /// What the latest measurement saw of it; nothing without one.
    pub seen: Sighting,
}

impl Subject {
    /// How many pixels the sphere bounding the subject spans across the
    /// view, for a view whose focal length is `focal_px`
    /// ([`ViewFacts::focal_px`]): half a turn's worth from inside it.
    pub fn extent_px(&self, focal_px: f32) -> f32 {
        let half = self.size * 0.5;
        2.0 * asin((half / self.center_distance.max(f32::MIN_POSITIVE)).min(1.0)) * focal_px
    }
}

/// What the latest measurement of the depth saw beyond its subjects.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Seen {
    /// How many frames before this one it was taken.
    pub age: u32,
    /// The share of the screen showing sky.
    pub sky: f32,
    /// The surfaces no subject holds.
    pub other: Sighting,
}

/// What the camera sees this frame: measured on the CPU from the camera and
/// the world, and from the latest of the frame depths read back from the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewFacts {
    /// Pixels one radian spans at the center of the view. Something `size`
    /// meters across `distance` meters away covers about
    /// `size * focal_px / distance` pixels ([`Self::pixels`]).
    pub focal_px: f32,
    /// The size the scene renders at, in pixels.
    pub render_size: [u32; 2],
    /// The camera's height over the planet's ground under it, in meters;
    /// `None` without a planet.
    pub ground_height: Option<f32>,
    /// The camera's height over the planet's sea, in meters, negative under
    /// it; `None` without a sea.
    pub sea_height: Option<f32>,
    /// How far the view moved since the last frame, in pixels at its center:
    /// its turn, plus its move against the nearest still surface the latest
    /// measurement saw ([`Self::nearest_still`]). 0 for a view that does not
    /// continue the last.
    pub motion_px: f32,
    /// The first directional light's light on level ground under the camera,
    /// in the light's own units: its color's luminance by the sine of its
    /// height over the horizon there.
    pub direct_light: f32,
    /// What the latest measurement saw beyond the subjects; `None` before one
    /// arrives and after a view that does not continue the last.
    pub seen: Option<Seen>,
    /// The subjects in the world this frame, each in its own slot: the planet
    /// first, then each water surface, then the prop.
    pub subjects: [Option<Subject>; VIEW_SUBJECTS],
}

impl ViewFacts {
    /// How many pixels something `size` meters across covers `distance`
    /// meters from the camera, at the center of the view.
    pub fn pixels(&self, size: f32, distance: f32) -> f32 {
        size * self.focal_px / distance.max(f32::MIN_POSITIVE)
    }

    /// The subjects in the world this frame.
    pub fn subjects(&self) -> impl Iterator<Item = &Subject> {
        self.subjects.iter().flatten()
    }

    /// The share of the screen the subjects of `kind` covered.
    pub fn share(&self, kind: SubjectKind) -> f32 {
        self.subjects()
            .filter(|s| s.kind == kind)
            .map(|s| s.seen.coverage)
            .sum()
    }

    /// How far the nearest surface the latest measurement saw lay from the
    /// camera, in meters, whatever held it; `None` when it saw none.
    pub fn nearest(&self) -> Option<f32> {
        self.nearest_of(|_| true)
    }

    /// The same, leaving out the prop, which may travel with the camera: the
    /// nearest surface the view moves against.
    pub fn nearest_still(&self) -> Option<f32> {
        self.nearest_of(|kind| kind != SubjectKind::Prop)
    }

    fn nearest_of(&self, counts: impl Fn(SubjectKind) -> bool) -> Option<f32> {
        let seen = self.seen?;
        self.subjects()
            .filter(|s| counts(s.kind))
            .filter_map(|s| s.seen.nearest)
            .chain(seen.other.nearest)
            .reduce(f32::min)
    }
}

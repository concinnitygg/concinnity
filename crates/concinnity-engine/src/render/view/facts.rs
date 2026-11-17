//! The facts a frame's view measured, as passes and reports read them.

use super::screen::ScreenRect;
use crate::gfx::render_types::{MAX_LEVEL_LAYERS, ViewSampleGrid};
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
    /// `None` when none of it was seen. A subject a measurement missed keeps
    /// the nearest an earlier one saw until as many in a row have missed it
    /// as the samples take to land on a band of it a pixel thin
    /// ([`ViewSampleGrid::sweep`]), so a sliver of it is not taken for
    /// hidden.
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
    /// How far it stands clear of the world's other surfaces, in meters: the
    /// gap between the prop's box and the highest the planet's ground reaches
    /// or any water surface, 0 where it meets one, lies within the ground's
    /// relief, or where the world has neither. 0 for the planet and the
    /// water, whose own relief lies beside them.
    pub clearance: f32,
    /// Whether its bounds meet the view, whether or not anything hides it.
    pub in_view: bool,
    /// The part of the screen its bounds can cover: the whole screen for the
    /// planet and the water, which can fill it.
    pub rect: ScreenRect,
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

/// One surface in view at the size the frame shows it: a subject, or the
/// surfaces no subject holds. Every consumer turns its own feature length
/// (an effect's reach, a blade's height, a wave's length) into pixels with
/// [`Self::pixels`], so one rule fits a pebble and a planet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detail {
    /// What it is; `None` for the surfaces no subject holds.
    pub kind: Option<SubjectKind>,
    /// How far its nearest visible part lies from the camera, in meters.
    pub nearest: f32,
    /// How far it stands clear of the world's other surfaces, in meters
    /// ([`Subject::clearance`]); 0 for the surfaces no subject holds.
    pub clearance: f32,
    /// The view's focal length in pixels ([`ViewFacts::focal_px`]).
    pub focal_px: f32,
    /// The part of the screen it can cover ([`Subject::rect`]); the whole
    /// screen for the surfaces no subject holds.
    pub rect: ScreenRect,
}

impl Detail {
    /// How many pixels something `length` meters long spans at its nearest
    /// visible part.
    pub fn pixels(&self, length: f32) -> f32 {
        length * self.focal_px / self.nearest.max(f32::MIN_POSITIVE)
    }
}

/// What the latest measurement of the depth saw beyond its subjects.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Seen {
    /// How many frames before this one it was taken.
    pub age: u32,
    /// How far the view moved since it was taken, in pixels at the view's
    /// center: each frame's [`ViewFacts::motion_px`] since, summed.
    pub drift_px: f32,
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
    /// The part of [`Self::motion_px`] the camera's travel makes: how far the
    /// nearest still surface shifted against the view. A turn moves every
    /// surface alike and uncovers nothing behind another; travel does.
    pub parallax_px: f32,
    /// How far, in pixels at the view's center, the fastest thing that moved
    /// in view since the last frame moved against the camera; 0 when nothing
    /// in view moved, and infinite for movement that cannot be placed (a
    /// change of pose, or more movers than are kept).
    pub mover_px: f32,
    /// How bright the auto-exposure shows the view's lit surfaces, against
    /// the target it shows a view's mean at with no bias
    /// ([`ExposureAdaptation::shown_light`](crate::gfx::auto_exposure::ExposureAdaptation::shown_light)):
    /// about 1 for an evenly lit view it has caught up with, under 1 for a
    /// night it biases down for or one too dark for its range. `None`
    /// without auto-exposure.
    pub shown_light: Option<f32>,
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

    /// Each surface in view at the size it shows, from the latest measurement
    /// while that still stands for the view, else from where the subjects lie.
    ///
    /// A subject in view counts at its nearest seen part while the measurement
    /// stands and saw it, or one of the measurements just before it did (see
    /// [`Sighting::nearest`]). A standing measurement that saw none of it leaves it
    /// out as hidden, unless it spans less than a sample cell, which the
    /// samples could miss. Without a standing measurement, a subject in view
    /// counts at its nearest point, seen or not. The surfaces no subject holds
    /// count where the measurement saw them, and at the camera itself when it
    /// no longer stands but saw some, or when nothing has been measured.
    pub fn details(&self) -> impl Iterator<Item = Detail> + '_ {
        let [width, height] = self.render_size;
        let cell = ViewSampleGrid::over(width, height).cell as f32;
        let standing = self.seen.filter(|seen| seen.drift_px <= cell);
        let focal_px = self.focal_px;
        let subjects = self.subjects().filter(|s| s.in_view).filter_map(move |s| {
            let nearest = match (standing, s.seen.nearest) {
                (Some(_), Some(seen)) => seen,
                (Some(_), None) if s.extent_px(focal_px) >= cell => return None,
                _ => s.distance,
            };
            Some(Detail {
                kind: Some(s.kind),
                nearest,
                clearance: s.clearance,
                focal_px,
                rect: s.rect,
            })
        });
        let other = match (self.seen, standing) {
            (None, _) => Some(0.0),
            (Some(_), Some(standing)) => standing.other.nearest,
            (Some(seen), None) => seen.other.nearest.map(|_| 0.0),
        };
        subjects.chain(other.map(|nearest| Detail {
            kind: None,
            nearest,
            clearance: 0.0,
            focal_px,
            rect: ScreenRect::FULL,
        }))
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

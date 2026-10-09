//! Planar reflection planning and math: which flat reflectors get a mirror
//! render, the camera mirrored across each reflector plane, and the part of the
//! screen each mirror render has to cover this frame.
//!
//! Backend-agnostic and pure (mirrors gfx/reflection_probe.rs). A reflective flat
//! surface (water, a glass pane) renders the scene a second time from the camera
//! reflected across its plane; the reflective surface then samples that render
//! projectively, at its own screen position. So only the reflector's own screen
//! footprint of a mirror render is ever read, which is what [`PlanarFramePlan`]
//! bounds: a plane whose reflectors are off screen renders nothing, and a visible
//! one renders only the rectangle its reflectors cover.
//!
//! Conventions match [`crate::gfx::projection`]: column-major storage
//! `m[col][row]`, a right-handed view looking down -z, and a perspective
//! projection mapping depth to [0, 1] (Metal / D3D clip space). A plane is
//! `[nx, ny, nz, d]` with `n` unit-length, satisfying `n . p + d = 0` for points
//! on it; `n . p + d > 0` is the side the normal points toward. Screen UVs and
//! pixel rectangles have a top-left origin, matching the fragment's screen UV.

mod mirror;
mod reflectors;
mod slots;
mod visibility;

pub use mirror::{PlanarMatrices, orient_plane_toward, planar_matrices};
pub use reflectors::{PlanarReflectors, pane_plane, water_plane};
pub use slots::{PlanarAssignment, assign_planar_slots, planar_pass_needed};
pub use visibility::{
    PixelRect, PlanarCrops, PlanarFramePlan, PlanarReflector, ReflectorHull, UvRect,
};

/// The engine-wide capacity ceiling for distinct reflection planes: water
/// surfaces and glass panes combined. Each plane is a scene re-render, so this
/// bounds the per-frame planar cost and the reserved mirror target VRAM;
/// reflectors past the active budget fall back to the box-projected probe cube.
/// This is the CAPACITY every backend sizes its mirror targets / ICB slots /
/// resolve SRVs against, so the three `planar::MAX_PLANAR_PLANES` alias it and
/// stay in lockstep by construction. The per-frame budget passed to
/// `assign_planar_slots` can be lower (scaled down under a quality preset / GPU
/// tier) but never higher.
pub const MAX_PLANAR_PLANES: usize = 4;

/// Distance a mirror's clip plane is pushed toward the kept (camera) side, so
/// geometry exactly on the surface is not lost to near-plane precision.
pub const PLANAR_CLIP_BIAS: f32 = 0.02;

/// Texels a mirror's crop is grown by on every side, covering the bilinear
/// footprint of the reflector's lookup.
pub const PLANAR_CROP_MARGIN: u32 = 2;

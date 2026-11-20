// The engine's own rigid-body simulation. A step runs `broadphase`,
// `collide`, `contact` (carrying impulses over), `island`, then `solver`, with
// `joint` solved inside it and `ccd` after it; `world` owns the storage and the
// order. `query` and `character` read the same storage without stepping.
// `sensor` and `impact` report what a step found, off stages that run anyway.

mod aabb;
mod body;
mod broadphase;
mod ccd;
mod character;
mod collide;
mod config;
mod contact;
mod coupling;
mod fluid;
mod impact;
mod island;
mod joint;
mod mass;
mod math;
mod narrow;
mod query;
mod scene;
mod sensor;
mod solver;
mod world;

pub use character::CharacterCapsule;
pub use config::{AirFalloff, SimConfig};
pub use fluid::{FluidLoad, FluidPiece, FluidShape, MAX_FLUID_PIECES, Submerged, submerged};
pub(crate) use query::ray_hit_distance;
pub use query::{ShapeCast, ShapeCastHit};
pub use world::Simulation;

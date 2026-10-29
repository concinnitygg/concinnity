// The engine's own rigid-body simulation. It lives beside the vocabulary it
// is driven through because that vocabulary was written for it; nothing here
// names a third party.
//
// The pipeline a step runs is the conventional one, and each stage is its own
// module: refresh bounds and sweep for candidate pairs (`broadphase`), build
// contact manifolds for those pairs (`collide`), match them against last
// step's manifolds so impulses carry over (`contact`), group them into islands
// so a settled group can stop being simulated (`island`), then solve
// (`solver`). `world` owns the storage and drives the stages in order.
//
// `joint` sits inside that solve rather than after it. A constraint a caller
// authored between two bodies is held by the same substepped impulses the
// contacts are, because a joint solved separately would fight whatever the
// contacts had just decided.
//
// `ccd` sits after that solve. A body whose step carried it further than the
// geometry it was heading for is thick would be missed by a contact test that
// only ever looks at where the step began and ended, so the ones moving fast
// enough for that are swept along their own path and stopped where they first
// met something.
//
// `query` sits beside that pipeline rather than in it: it reads the same
// storage and the same broad-phase order, but it neither steps nor mutates,
// so a ray or a sweep can be asked between steps or during one. `character`
// sits on that same side: a capsule driven by a desired translation is
// resolved out of sweeps, not out of the solver.
//
// `sensor` and `impact` are what a step says back. Both are read off stages
// that had to run anyway -- the sweep's pairs and the solved manifolds -- so a
// world nobody is listening to pays for them in a threshold test and nothing
// more.

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
pub use config::SimConfig;
pub use fluid::{FluidLoad, FluidPiece, FluidShape, MAX_FLUID_PIECES, Submerged, submerged};
pub(crate) use query::ray_hit_distance;
pub use query::{ShapeCast, ShapeCastHit};
pub use world::Simulation;

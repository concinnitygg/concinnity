// concinnity-physics/src/sim/scene.rs
//
// The three things a query has to be handed together: the bodies, the order
// they are swept in, and the height grids some of them stand for.
//
// It is one struct rather than three arguments because every query and the
// character controller need all three, and threading them separately grows a
// parameter list at each layer without ever giving a caller a choice about
// which to pass.

use concinnity_memory::Pool;

use super::body::Body;
use super::broadphase::SweepPrune;
use super::collide::heightfield::Heightfields;

/// What a query reads. Borrowed, never owned: a query changes nothing.
#[derive(Clone, Copy)]
pub(crate) struct Scene<'a> {
    pub(crate) bodies: &'a Pool<Body>,
    pub(crate) broadphase: &'a SweepPrune,
    pub(crate) fields: &'a Heightfields,
}

// The bodies, their sweep order and the height grids, handed to every query
// and the character controller together.

use concinnity_core::memory::Pool;

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

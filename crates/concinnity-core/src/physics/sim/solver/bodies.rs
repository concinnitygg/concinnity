// The state a step holds each body in, and how several islands hold the array
// of it at once.
//
// The array is indexed by pool slot, so an island's bodies are scattered
// through it rather than sitting in a run of their own. There is no slice to
// split, so what makes it shareable is the partition instead: union-find puts
// every slot the step moves in exactly one island, a chunk of work holds a
// disjoint set of islands, and an entry two chunks both reach is one the step
// cannot move -- read by both, written by neither.
//
// That last half is what `apply_impulse` declining an immovable body buys.
// Zero inverse mass already made the write a no-op arithmetically; declining
// it makes it a no-op in memory, which is what turns the floor a hundred
// separate stacks lean on into shared, read-only state.

use core::marker::PhantomData;

use crate::physics::sim::body::Body;
use crate::physics::sim::math::{Mat3, Quat, Vec3};

/// One body's state for the duration of a step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SolverBody {
    pub(crate) linear_velocity: Vec3,
    pub(crate) angular_velocity: Vec3,
    pub(crate) position: Vec3,
    pub(crate) rotation: Quat,
    /// How far the body has moved since the step began, which is what the
    /// contact separation is re-measured against between substeps.
    pub(crate) delta_position: Vec3,
    pub(crate) delta_rotation: Quat,
    pub(super) start_rotation_conjugate: Quat,
    pub(super) start_position: Vec3,
    pub(crate) inv_mass: f32,
    pub(super) inv_inertia_local: Vec3,
    pub(crate) inv_inertia: Mat3,
    pub(crate) gravity_scale: f32,
    pub(crate) damping: f32,
    /// Whether this step moves the body at all. A static or sleeping body is
    /// present so contacts can reference its pose, and immovable to the solve.
    pub(crate) simulated: bool,
}

impl SolverBody {
    /// The state an unoccupied slot holds: present, and moved by nothing.
    pub(crate) const IMMOVABLE: SolverBody = SolverBody {
        linear_velocity: Vec3::ZERO,
        angular_velocity: Vec3::ZERO,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        delta_position: Vec3::ZERO,
        delta_rotation: Quat::IDENTITY,
        start_rotation_conjugate: Quat::IDENTITY,
        start_position: Vec3::ZERO,
        inv_mass: 0.0,
        inv_inertia_local: Vec3::ZERO,
        inv_inertia: Mat3::ZERO,
        gravity_scale: 0.0,
        damping: 0.0,
        simulated: false,
    };

    pub(crate) fn from_body(body: &Body) -> Self {
        let simulated = body.is_simulated();
        SolverBody {
            linear_velocity: body.linear_velocity,
            angular_velocity: body.angular_velocity,
            position: body.position,
            rotation: body.orientation,
            delta_position: Vec3::ZERO,
            delta_rotation: Quat::IDENTITY,
            start_rotation_conjugate: body.orientation.conjugate(),
            start_position: body.position,
            inv_mass: if simulated { body.inv_mass } else { 0.0 },
            inv_inertia_local: if simulated {
                body.inv_inertia_local
            } else {
                Vec3::ZERO
            },
            inv_inertia: if simulated {
                body.inv_inertia_world()
            } else {
                Mat3::ZERO
            },
            gravity_scale: body.gravity_scale,
            damping: body.damping,
            simulated,
        }
    }

    pub(crate) fn velocity_at(&self, r: Vec3) -> Vec3 {
        self.linear_velocity + self.angular_velocity.cross(r)
    }

    /// Take an impulse through an arm from the centre of mass.
    ///
    /// A body the step cannot move is left alone rather than added to by zero:
    /// its inverse mass and inertia are both zero, so the arithmetic changed
    /// nothing, and not touching it is what lets every island leaning on the
    /// same wall solve at once.
    pub(crate) fn apply_impulse(&mut self, impulse: Vec3, r: Vec3) {
        if !self.simulated {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        self.angular_velocity += self.inv_inertia.mul_vec3(r.cross(impulse));
    }

    /// A pure couple: no lever arm, so nothing but the spin changes.
    pub(crate) fn apply_angular_impulse(&mut self, impulse: Vec3) {
        if !self.simulated {
            return;
        }
        self.angular_velocity += self.inv_inertia.mul_vec3(impulse);
    }

    pub(crate) fn integrate_position(&mut self, h: f32) {
        self.position += self.linear_velocity * h;
        self.rotation = self.rotation.integrate(self.angular_velocity, h);
        self.delta_position = self.position - self.start_position;
        self.delta_rotation = self.rotation.mul(self.start_rotation_conjugate);
        if self.inv_inertia_local != Vec3::ZERO {
            self.inv_inertia = Mat3::diagonal_conjugated(self.rotation, self.inv_inertia_local);
        }
    }
}

/// A handle on the step's body states that several chunks of work may hold at
/// the same time.
///
/// Copying one is what the island partition licenses, so the copy is made
/// through [`Bodies::share`] rather than by deriving `Clone`: every duplicate
/// exists because a caller has established that the two holders reach disjoint
/// slots.
pub(crate) struct Bodies<'a> {
    at: *mut SolverBody,
    len: usize,
    owner: PhantomData<&'a mut [SolverBody]>,
}

// SAFETY: `Bodies` is a `&mut [SolverBody]` that has been split by a rule the
// type system cannot express. It carries no interior mutability of its own and
// hands out no reference that outlives the borrow it was made from, so sending
// one to another thread is sound exactly when that thread reaches slots no
// other holder writes -- which is the invariant `share` documents and the
// solver's partition establishes.
unsafe impl Send for Bodies<'_> {}

impl Default for Bodies<'_> {
    fn default() -> Self {
        Bodies {
            at: core::ptr::null_mut(),
            len: 0,
            owner: PhantomData,
        }
    }
}

impl<'a> Bodies<'a> {
    pub(crate) fn new(bodies: &'a mut [SolverBody]) -> Self {
        Bodies {
            at: bodies.as_mut_ptr(),
            len: bodies.len(),
            owner: PhantomData,
        }
    }

    /// Another handle on the same bodies.
    ///
    /// # Safety
    ///
    /// Every slot the new handle is used to write must be one no other live
    /// handle writes or reads. The solver satisfies this by giving each chunk
    /// a disjoint set of islands: a slot the step moves belongs to exactly one
    /// island, and the slots two chunks share are ones neither writes.
    pub(crate) unsafe fn share(&self) -> Bodies<'a> {
        Bodies {
            at: self.at,
            len: self.len,
            owner: PhantomData,
        }
    }

    pub(crate) fn get(&self, slot: u32) -> &SolverBody {
        assert!(
            (slot as usize) < self.len,
            "body slot {slot} is out of range"
        );
        // SAFETY: the index is in range, the pointer came from a live
        // `&mut [SolverBody]` this handle borrows for `'a`, and the returned
        // shared reference cannot outlive that borrow. Whoever else holds this
        // slot at the same time only reads it, per `share`'s contract.
        unsafe { &*self.at.add(slot as usize) }
    }

    pub(crate) fn get_mut(&mut self, slot: u32) -> &mut SolverBody {
        assert!(
            (slot as usize) < self.len,
            "body slot {slot} is out of range"
        );
        // SAFETY: as above, and the returned exclusive reference borrows
        // `self`, so one handle never hands out two at once. Two handles reach
        // this slot only if `share`'s contract was met, which means no other
        // holder touches it.
        unsafe { &mut *self.at.add(slot as usize) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::vec3;

    fn movable() -> SolverBody {
        SolverBody {
            inv_mass: 2.0,
            inv_inertia: Mat3::IDENTITY,
            simulated: true,
            ..SolverBody::IMMOVABLE
        }
    }

    #[test]
    fn an_immovable_body_declines_every_impulse() {
        let mut body = SolverBody::IMMOVABLE;
        body.apply_impulse(vec3(5.0, 5.0, 5.0), Vec3::X);
        body.apply_angular_impulse(vec3(5.0, 5.0, 5.0));
        assert_eq!(body.linear_velocity, Vec3::ZERO);
        assert_eq!(body.angular_velocity, Vec3::ZERO);
    }

    #[test]
    fn a_movable_body_takes_it() {
        let mut body = movable();
        body.apply_impulse(vec3(1.0, 0.0, 0.0), Vec3::ZERO);
        assert_eq!(body.linear_velocity, vec3(2.0, 0.0, 0.0));
        body.apply_angular_impulse(vec3(0.0, 3.0, 0.0));
        assert_eq!(body.angular_velocity, vec3(0.0, 3.0, 0.0));
    }

    // The whole point of the shared handle: two of them writing different
    // slots is one array being filled in, not two views fighting.
    #[test]
    fn two_handles_write_disjoint_slots() {
        let mut storage = [movable(), movable(), movable()];
        let mut first = Bodies::new(&mut storage);
        // SAFETY: the two handles below are used on slots 0 and 2, which are
        // disjoint, and slot 1 is touched by neither.
        let mut second = unsafe { first.share() };
        first.get_mut(0).linear_velocity = Vec3::X;
        second.get_mut(2).linear_velocity = Vec3::Y;
        assert_eq!(first.get(0).linear_velocity, Vec3::X);
        assert_eq!(first.get(1).linear_velocity, Vec3::ZERO);
        assert_eq!(second.get(2).linear_velocity, Vec3::Y);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn a_slot_past_the_end_is_refused() {
        let mut storage = [movable()];
        let bodies = Bodies::new(&mut storage);
        bodies.get(4);
    }

    #[test]
    fn an_empty_handle_holds_nothing() {
        let bodies = Bodies::default();
        assert_eq!(bodies.len, 0);
    }
}

// A resource slot for a value that systems take out for the length of a step
// and put back, so the value and the `PipelineContext` are never borrowed
// together.

use super::Resources;

/// One value parked in the world's resources between the steps that use it.
/// `None` only while a step has it taken.
///
/// ```rust
/// # use concinnity_core::ecs::{Parked, Resources};
/// let mut resources = Resources::new();
/// Parked::put(&mut resources, 7u32);
/// assert_eq!(Parked::<u32>::with(&mut resources, |n| *n + 1), Some(8));
/// assert_eq!(Parked::<u32>::take(&mut resources), Some(7));
/// assert_eq!(Parked::<u32>::take(&mut resources), None);
/// ```
pub struct Parked<T>(pub Option<T>);

impl<T: Send + 'static> Parked<T> {
    /// A slot holding `value`.
    pub fn new(value: T) -> Self {
        Self(Some(value))
    }

    /// Take the parked value for the duration of one step.
    pub fn take(resources: &mut Resources) -> Option<T> {
        resources.get_mut::<Self>()?.0.take()
    }

    /// Park `value` again, reusing the slot when one exists.
    pub fn put(resources: &mut Resources, value: T) {
        match resources.get_mut::<Self>() {
            Some(slot) => slot.0 = Some(value),
            None => {
                resources.insert(Self::new(value));
            }
        }
    }

    /// Borrow the parked value, if one is parked.
    pub fn get(resources: &Resources) -> Option<&T> {
        resources.get::<Self>()?.0.as_ref()
    }

    /// Run `f` against the parked value in place. `None` when nothing is
    /// parked.
    pub fn with<R>(resources: &mut Resources, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let slot = resources.get_mut::<Self>()?;
        slot.0.as_mut().map(f)
    }
}

impl<T> core::fmt::Debug for Parked<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Parked").field(&self.0.is_some()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_taken_value_parks_back_in_its_slot() {
        let mut resources = Resources::new();
        assert_eq!(Parked::<u8>::take(&mut resources), None);
        Parked::put(&mut resources, 3u8);
        let value = Parked::<u8>::take(&mut resources);
        assert_eq!(value, Some(3));
        assert!(Parked::<u8>::get(&resources).is_none(), "taken");
        Parked::put(&mut resources, 4u8);
        assert_eq!(Parked::<u8>::get(&resources), Some(&4));
    }

    #[test]
    fn with_runs_only_against_a_parked_value() {
        let mut resources = Resources::new();
        assert_eq!(Parked::<u8>::with(&mut resources, |v| *v), None);
        Parked::put(&mut resources, 1u8);
        assert_eq!(
            Parked::<u8>::with(&mut resources, |v| {
                *v += 1;
                *v
            }),
            Some(2)
        );
        assert_eq!(Parked::<u8>::get(&resources), Some(&2));
    }
}

//! The world an application is assembled from.

use concinnity_core::ecs::RuntimeComponent;

// The runtime's world when there is a runtime, its data half when there is
// not. Both carry the same components; only the std one can run them.
#[cfg(feature = "std")]
pub(crate) type Inner = concinnity_engine::ecs::World;
#[cfg(not(feature = "std"))]
pub(crate) type Inner = concinnity_core::ecs::World;

/// A world: the components an application is built from.
///
/// Components are added one at a time with [`add_component`](World::add_component),
/// or compiled in bulk from authored assets by the `cook` module. An `App`
/// runs the result.
#[derive(Default)]
pub struct World {
    inner: Inner,
}

impl core::fmt::Debug for World {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("World").finish_non_exhaustive()
    }
}

impl World {
    /// An empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one component to the world.
    ///
    /// Only a runtime component can be added. A build-only asset (`Prefab`,
    /// `MainMenu`, `CharacterSchema`, ...) is consumed by the cook and expanded
    /// into the components it stands for, so it is rejected here at compile
    /// time; declare it in a world and build instead.
    ///
    /// ```
    /// # use concinnity::World;
    /// # use concinnity::assets::TextLabel;
    /// let mut world = World::new();
    /// world.add_component(TextLabel {
    ///     content: "Hello, world!".to_string(),
    ///     ..Default::default()
    /// });
    /// ```
    pub fn add_component<C: RuntimeComponent>(&mut self, component: C) {
        self.inner.add_component(component);
    }

    #[cfg(feature = "cook")]
    pub(crate) fn from_inner(inner: Inner) -> Self {
        Self { inner }
    }

    #[cfg(feature = "std")]
    pub(crate) fn into_inner(self) -> Inner {
        self.inner
    }

    #[cfg(test)]
    pub(crate) fn inner(&self) -> &Inner {
        &self.inner
    }
}

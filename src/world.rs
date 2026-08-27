//! The world an application is assembled from.

use concinnity_core::ecs::RuntimeComponent;

// One world on both tiers: it carries the components and the systems built over
// them, and needs no operating system to do either. What differs is what a tier
// has to put in it -- the std build's `App` starts it against the engine's
// system table.
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
    /// Only a runtime component can be added, which is every type in
    /// [`components`](crate::components). A build-only asset (`Prefab`,
    /// `MainMenu`, `CharacterSchema`, ...) stands for several components rather
    /// than being one, so it is rejected here at compile time; declare it
    /// through the `cook` module instead.
    #[cfg_attr(feature = "cook", doc = " See [`cook`](mod@crate::cook).")]
    ///
    /// ```
    /// # use concinnity::World;
    /// # use concinnity::components::TextLabel;
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

    pub(crate) fn into_inner(self) -> Inner {
        self.inner
    }

    #[cfg(test)]
    pub(crate) fn inner(&self) -> &Inner {
        &self.inner
    }
}

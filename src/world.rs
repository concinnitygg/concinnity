//! The world an application is assembled from.

use alloc::vec::Vec;

use concinnity_core::components::{Material, ProceduralMesh};
use concinnity_core::ecs::RuntimeComponent;

use crate::{EnvironmentMapHandle, MaterialHandle, MeshHandle};

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

    /// Add a mesh with its baked geometry payload, returning the handle a
    /// [`Prop`](crate::components::Prop) references it by.
    ///
    /// The payload comes from [`bake::procedural_mesh`](crate::bake::procedural_mesh)
    /// (or, through the `cook` module, from a compiled world). Handles count up
    /// in the order meshes are added.
    pub fn add_mesh(&mut self, mesh: ProceduralMesh, payload: Vec<u8>) -> MeshHandle {
        self.inner.add_mesh(mesh, payload)
    }

    /// Add a material, returning the handle a
    /// [`Prop`](crate::components::Prop) references it by. The value's fields
    /// are clamped into their valid ranges on the way in, exactly as the
    /// `cook` module clamps an authored material.
    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        self.inner.add_material(material)
    }

    /// Add a baked image-based-lighting payload, from
    /// [`bake::environment_map`](crate::bake::environment_map). The renderer
    /// lights with the map at handle 0.
    pub fn add_environment_map(&mut self, payload: Vec<u8>) -> EnvironmentMapHandle {
        self.inner.add_environment_map(payload)
    }

    // Only the cook module compiles a core world it then wraps; the raw path
    // starts from `World::new` and never converts.
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

    // Only the cook-vs-bake parity oracle starts a world to compare the two,
    // and it is the one test that needs the inner world mutably.
    #[cfg(all(test, feature = "cook"))]
    pub(crate) fn inner_mut(&mut self) -> &mut Inner {
        &mut self.inner
    }
}

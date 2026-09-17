//! The world an application is assembled from.

use concinnity_core::components::Material;
use concinnity_core::ecs::RuntimeComponent;

use crate::bake::{EnvironmentMapPayload, FontPayload, MeshPayload};
use crate::system::{ComponentSlot, Entity, Phase, System};

use crate::{CnError, EnvironmentMapHandle, FontHandle, MaterialHandle, MeshHandle};

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

    /// Allocate an entity that holds no components yet, to be filled with
    /// [`insert`](World::insert).
    ///
    /// This and `insert` are how a world is seeded with an application's own
    /// component types, which are not part of the
    /// [`components`](crate::components) vocabulary and so cannot go through
    /// [`add_component`](World::add_component). See
    /// [`declare_components!`](crate::declare_components).
    ///
    /// ```
    /// # use concinnity::{World, declare_components};
    /// # #[derive(Debug)]
    /// # struct Health(u32);
    /// # declare_components!(Health);
    /// let mut world = World::new();
    /// let player = world.spawn();
    /// world.insert(player, Health(100));
    /// ```
    pub fn spawn(&mut self) -> Entity {
        self.inner.spawn()
    }

    /// Give an existing entity one more component, of any type with a column:
    /// a [`components`](crate::components) type, or one of the application's
    /// own.
    ///
    /// `false`, leaving the world unchanged, when the entity is dead or already
    /// holds this component type.
    pub fn insert<C: ComponentSlot>(&mut self, entity: Entity, component: C) -> bool {
        self.inner.insert(entity, component)
    }

    /// Add a component on an entity of its own, returning that entity so more
    /// can be added to it. The one-call form of [`spawn`](World::spawn)
    /// followed by [`insert`](World::insert).
    ///
    /// Each call makes a *new* entity, which is the difference from
    /// [`insert`](World::insert): to put several components on one thing, push
    /// the first and insert the rest onto the entity that comes back.
    ///
    /// ```
    /// # use concinnity::{World, declare_components};
    /// # use concinnity::system::Entity;
    /// # #[derive(Debug)]
    /// # struct Health(u32);
    /// # #[derive(Debug)]
    /// # struct Faction(&'static str);
    /// # declare_components!(Health, Faction);
    /// let mut world = World::new();
    ///
    /// // One entity, two components.
    /// let player: Entity = world.push(Health(100));
    /// world.insert(player, Faction("blue"));
    ///
    /// // A second push is a second entity, not a second component on the first.
    /// let enemy = world.push(Health(50));
    /// assert_ne!(player, enemy);
    /// ```
    ///
    /// Unlike [`add_component`](World::add_component), which takes only the
    /// [`components`](crate::components) vocabulary, this takes any type with a
    /// column: a vocabulary type, or one of the application's own.
    pub fn push<C: ComponentSlot>(&mut self, component: C) -> Entity {
        self.inner.push(component)
    }

    /// Register a system on the world, to run in `phase` under `name`.
    ///
    /// This is how code of your own joins the tick the engine's systems run on.
    /// See [`system`](mod@crate::system) for what a system is and how to write
    /// one.
    ///
    /// Registration order is run order within a phase, and every engine system
    /// in a phase runs before every system registered into it, so registering
    /// never reorders the engine's own tick. `name` is what the profile and the
    /// log address the system by; it cannot repeat an engine system's name or an
    /// earlier registration's, and a repeat makes [`App::run`](crate::App::run)
    /// return an error.
    ///
    /// ```
    /// # use concinnity::system::{Phase, PipelineContext, StepResult, System};
    /// # use concinnity::World;
    /// # #[derive(Debug)]
    /// # struct Ticker;
    /// # impl System for Ticker {
    /// #     fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult { StepResult::Continue }
    /// # }
    /// let mut world = World::new();
    /// world.add_system(Phase::Late, "Ticker", Ticker);
    /// ```
    pub fn add_system<S: System>(&mut self, phase: Phase, name: &'static str, system: S) {
        self.inner.add_system(phase, name, system);
    }

    /// Add a baked mesh payload, returning the handle a
    /// [`Prop`](crate::components::Prop) references it by.
    ///
    /// The payload is what [`bake::procedural_mesh`](crate::bake::procedural_mesh)
    /// generated from a [`ProceduralMesh`](crate::components::ProceduralMesh),
    /// which the world also keeps, or what [`bake::mesh`](crate::bake::mesh)
    /// packed from a raw [`bake::Mesh`](crate::bake::Mesh)'s vertices.
    /// Handles count up in the order meshes are added.
    ///
    /// Errors once the world has minted
    /// [`AssetId::MINTED_CAPACITY`](crate::AssetId::MINTED_CAPACITY)
    /// names.
    pub fn add_mesh(&mut self, payload: MeshPayload) -> Result<MeshHandle, CnError> {
        self.inner.add_mesh(payload)
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
    ///
    /// Only an environment-map payload fits; a mesh's does not:
    ///
    /// ```compile_fail
    /// # use concinnity::{World, bake};
    /// # fn main() -> Result<(), concinnity::Error> {
    /// let mut world = World::new();
    /// let geometry = bake::mesh(&bake::Mesh::default())?;
    /// world.add_environment_map(geometry);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_environment_map(&mut self, payload: EnvironmentMapPayload) -> EnvironmentMapHandle {
        self.inner.add_environment_map(payload.into_bytes())
    }

    /// Add a baked glyph atlas, from [`bake::font`](crate::bake::font),
    /// returning the handle a [`TextLabel`](crate::components::TextLabel)
    /// references it by.
    pub fn add_font(&mut self, payload: FontPayload) -> FontHandle {
        self.inner.add_font(payload.into_bytes())
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

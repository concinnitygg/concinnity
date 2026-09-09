// src/ecs/declare.rs
//
// The macro an application declares its own component types with.
//
// The engine's registry is a closed list, and stays one: its entries have blob
// discriminants, authoring names, and a column each in the storage struct. A
// type declared here has none of that. It gets a discriminant past the last
// registered one, a column in `ExtColumns`, and the `ComponentSlot` impl that
// makes the two reachable -- which is the whole of what the ECS asks of a
// component type.
//
// Discriminants come from list position, exactly as the registry's do, so
// nothing is hand-numbered. That is also why one invocation has to cover an
// application's whole set: a second would start counting from the same base.

/// The first discriminant no registered component uses, and so where a
/// [`declare_components!`](crate::declare_components) list starts counting.
///
/// It moves whenever the engine's registry grows. Nothing persists a component
/// type declared outside the engine, so nothing has to be migrated when it
/// does; what shrinks is how many are left before the
/// [`ComponentId::MAX`](crate::ecs::ComponentId::MAX) ceiling.
pub const EXTENSION_COMPONENT_BASE: u8 = crate::ecs::ComponentTag::COUNT;

/// Declare an application's own component types, so a world can store them and
/// a system can query them.
///
/// Each type gets a column and a discriminant, which is what
/// [`ComponentSlot`](crate::ecs::ComponentSlot) needs, so every component
/// operation works on it: `push`, `insert`, `get`, `get_mut`, `remove`,
/// `query`, `query_mut`, `join2`, `changed_rows`, and the sweep a `despawn`
/// runs. A type declared here is runtime-only: it has no authoring name and no
/// blob record, so it cannot be written in a world file or survive a save.
///
/// The types are numbered by position, from
/// [`EXTENSION_COMPONENT_BASE`]. **One invocation covers an application's whole
/// set**: a second would number its own list from the same base and collide,
/// which the storage refuses the first time both columns are reached.
///
/// Each type must be `Debug + Send + Sync + 'static`, the same bounds the
/// registered components carry.
///
/// ```
/// use concinnity_core::declare_components;
/// use concinnity_core::ecs::World;
///
/// #[derive(Debug)]
/// struct Health(u32);
///
/// #[derive(Debug)]
/// struct Faction(&'static str);
///
/// declare_components!(Health, Faction);
///
/// let mut world = World::new();
/// let player = world.spawn();
/// world.insert(player, Health(100));
/// world.insert(player, Faction("blue"));
///
/// assert_eq!(world.get::<Health>(player).map(|h| h.0), Some(100));
/// ```
#[macro_export]
macro_rules! declare_components {
    ( $( $ty:ty ),* $(,)? ) => {
        $crate::__cn_declare_component!(0u8; $( $ty, )*);
    };
}

/// Internal: one list element and the index it takes, then the rest.
#[macro_export]
#[doc(hidden)]
macro_rules! __cn_declare_component {
    ( $index:expr; ) => {};
    ( $index:expr; $ty:ty, $( $rest:ty, )* ) => {
        impl $crate::ecs::ComponentSlot for $ty {
            const DISCRIMINANT: u8 = $crate::ecs::EXTENSION_COMPONENT_BASE + $index;

            fn column(
                storage: &$crate::ecs::ComponentStorage,
            ) -> ::core::option::Option<&$crate::ecs::Column<Self>> {
                storage
                    .ext
                    .column::<Self>(<Self as $crate::ecs::ComponentSlot>::DISCRIMINANT)
            }

            fn column_mut(
                storage: &mut $crate::ecs::ComponentStorage,
            ) -> &mut $crate::ecs::Column<Self> {
                storage
                    .ext
                    .column_mut::<Self>(<Self as $crate::ecs::ComponentSlot>::DISCRIMINANT)
            }
        }

        // The ComponentMask is a u128, so a discriminant past 127 would alias
        // another component's bit. Fail at the declaration instead.
        const _: () = assert!(
            <$ty as $crate::ecs::ComponentSlot>::DISCRIMINANT <= $crate::ecs::ComponentId::MAX,
            "too many component types: the registry's plus this list's exceed the 128 a ComponentMask holds",
        );

        $crate::__cn_declare_component!($index + 1u8; $( $rest, )*);
    };
}
